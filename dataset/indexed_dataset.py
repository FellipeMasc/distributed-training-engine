"""Megatron-compatible IndexedDataset binary format (.bin/.idx).

Adapted from Megatron-LM megatron/core/datasets/indexed_dataset.py.
Stripped of S3, MSC, and other cloud-storage dependencies to be self-contained.

The .idx file layout:
    - 9-byte header  (b"MMIDIDX\\x00\\x00")
    - 8-byte version (uint64, always 1)
    - 1-byte dtype code
    - 8-byte sequence count
    - 8-byte document count
    - sequence_lengths   (int32  × sequence_count)
    - sequence_pointers  (int64  × sequence_count)   byte offsets into .bin
    - document_indices   (int64  × document_count)    sequence-index boundaries

The .bin file is a flat concatenation of token-id arrays in the chosen dtype.
"""

import gc
import logging
import os
import shutil
import struct
import time
from collections.abc import Iterable
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union

import numpy
from sympy import sequence
import torch

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """NumPy dtype ↔ integer code mapping used by the .idx file."""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(vocab_size: Optional[int]) -> Type[numpy.number]:
        if vocab_size is not None and vocab_size < 65500:
            return numpy.uint16
        return numpy.int32


# ---------------------------------------------------------------------------
# Index writer
# ---------------------------------------------------------------------------

class _IndexWriter:
    """Writes the .idx file."""

    def __init__(self, idx_path: str, dtype: Type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        self.idx_writer = open(self.idx_path, "wb")
        self.idx_writer.write(_INDEX_HEADER)
        self.idx_writer.write(struct.pack("<Q", 1))
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.idx_writer.close()
        return None

    def write(
        self,
        sequence_lengths: Iterable[Union[int, numpy.integer]],
        sequence_modes: Optional[Iterable[Union[int, numpy.integer]]],
        document_indices: Iterable[Union[int, numpy.integer]],
    ) -> None:
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        self.idx_writer.write(numpy.array(sequence_lengths, dtype=numpy.int32).tobytes(order="C"))
        self.idx_writer.write(numpy.array(sequence_pointers, dtype=numpy.int64).tobytes(order="C"))
        self.idx_writer.write(numpy.array(document_indices, dtype=numpy.int64).tobytes(order="C"))

        if sequence_modes is not None:
            self.idx_writer.write(
                numpy.array(sequence_modes, dtype=numpy.int8).tobytes(order="C")
            )

    def _sequence_pointers(
        self, sequence_lengths: Iterable[Union[int, numpy.integer]]
    ) -> List[int]:
        itemsize = numpy.int64(DType.size(self.dtype))
        curr_ptr = numpy.int64(0)
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr.item())
            curr_ptr += length * itemsize
        return list_ptr


# ---------------------------------------------------------------------------
# Index reader
# ---------------------------------------------------------------------------

class _IndexReader:
    """Reads the .idx file via memory-map."""

    def __init__(self, idx_path: str, multimodal: bool = False) -> None:
        logger.info("Load the _IndexReader from %s", idx_path)

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )

        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        self.sequence_modes = None
        if multimodal:
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )

        logger.info("> total number of sequences: %d", len(self))
        logger.info("> total number of documents: %d", self.document_indices.shape[0] - 1)

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


# ---------------------------------------------------------------------------
# .bin reader (mmap only, no cloud storage)
# ---------------------------------------------------------------------------

class _MMapBinReader:
    """Memory-maps the .bin data file for fast random reads."""

    def __init__(self, bin_path: str) -> None:
        self._bin_file_reader = open(bin_path, mode="rb")
        self._bin_buffer_mmap = numpy.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        if self._bin_file_reader is not None:
            self._bin_file_reader.close()
        del self._bin_buffer_mmap
        del self._bin_file_reader


# ---------------------------------------------------------------------------
# IndexedDataset  (read-side)
# ---------------------------------------------------------------------------

class IndexedDataset(torch.utils.data.Dataset):
    """Reads a Megatron-format .bin/.idx pair as a torch Dataset.

    Args:
        path_prefix: The shared prefix for the .idx and .bin files.
        multimodal: Whether the dataset has per-sequence mode metadata.
    """

    def __init__(self, path_prefix: str, multimodal: bool = False) -> None:
        super().__init__()
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)
        assert os.path.exists(idx_path) and os.path.exists(bin_path), (
            f".idx or .bin not found at prefix {path_prefix}"
        )
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.bin_reader = _MMapBinReader(bin_path)
        self.index = _IndexReader(idx_path, multimodal)

        assert self.index.sequence_lengths.shape[0] == self.index.document_indices[-1]

    def __getstate__(self):
        return self.path_prefix, self.multimodal

    def __setstate__(self, state):
        path_prefix, multimodal = state
        self.__init__(path_prefix, multimodal)

    def __del__(self):
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, List[numpy.ndarray]]:
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = (
                self.index.sequence_modes[idx] if self.multimodal else None
            )
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError(f"Unexpected index type: {type(idx)}")

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """Retrieve a sub-range of a single sequence."""
        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = self.bin_reader.read(
            dtype=self.index.dtype, count=length, offset=sequence_pointer
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        return self.index.document_indices

    @staticmethod
    def exists(path_prefix: str) -> bool:
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )


class PackingDataset(IndexedDataset):
    
    def __init__(self, path_prefix: str, multimodal: bool = False, sequence_length: int = 512, tokenizer = None) -> None:
        super().__init__(path_prefix,multimodal)
        self.sequence_length = sequence_length
        self.cache_previous_sequences = {}
        self.tokenizer = tokenizer

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, List[numpy.ndarray]]:
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            labels = sequence[1:]
            labels = numpy.append(labels,128004)
            position_ids = torch.arange(sequence_length)
            return {
                "tokens" : torch.from_numpy(sequence.copy()),
                "labels" : torch.from_numpy(labels.copy()),
                "position_ids" : position_ids
            }
            # return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = (
                self.index.sequence_modes[idx] if self.multimodal else None
            )
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError(f"Unexpected index type: {type(idx)}")
    

# ---------------------------------------------------------------------------
# IndexedDatasetBuilder  (write-side)
# ---------------------------------------------------------------------------

class IndexedDatasetBuilder:
    """Builds an IndexedDataset (.bin + .idx pair) from tokenized documents.

    Usage:
        builder = IndexedDatasetBuilder("output.bin", dtype=numpy.uint16)
        builder.add_document(token_ids_list, sentence_lengths)
        builder.finalize("output.idx")
    """

    def __init__(
        self, bin_path: str, dtype: Type[numpy.number] = numpy.int32, multimodal: bool = False
    ) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths: List[int] = []
        self.document_indices: List[int] = [0]
        self.sequence_modes: Optional[List[int]] = [] if self.multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single sequence to the dataset."""
        np_array = numpy.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(
        self, tensor, lengths: List[int], modes: Optional[List[int]] = None
    ) -> None:
        """Add an entire document (list of token ids) with per-sentence lengths."""
        np_array = numpy.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * len(lengths))

    def end_document(self) -> None:
        """Mark the end of a document when using add_item."""
        self.document_indices.append(len(self.sequence_lengths))

    def add_index(self, path_prefix: str) -> None:
        """Merge another IndexedDataset into this builder (for partition merging)."""
        index = _IndexReader(get_idx_path(path_prefix))
        assert index.dtype == self.dtype

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)
        self.document_indices.extend((offset + index.document_indices)[1:])

        if self.multimodal:
            assert index.sequence_modes is not None
            self.sequence_modes.extend(index.sequence_modes)

        del index
        gc.collect()

        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """Flush the .bin file and write the .idx file."""
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_idx_path(path_prefix: str) -> str:
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    return path_prefix + ".bin"

"""Hand-rolled tiny ONNX models for the arcface-family tests.

The service's runtime dependency is onnxruntime alone; the `onnx` package
(graph-building helpers, generated protobuf classes) is deliberately NOT in
requirements.txt — pulling a second heavyweight dependency into every
docker image to serve a handful of test fixtures would invert the cost. An
.onnx file is just a serialized protobuf ModelProto, and the graphs these
tests need are ONE data-movement node (Flatten, default axis=1) with no
initializers and no attributes — small enough to encode by hand.

Flatten is chosen deliberately: it is pure data movement, so it has kernels
for EVERY tensor dtype (the fp16 fixture must actually RUN on the CPU EP),
and its output is the input verbatim — letting tests assert the exact
normalized/transposed blob ArcFaceEmbedder fed the graph, which pins the
NCHW-vs-NHWC branch and the dtype cast at value level, not just by shape.

Correctness of the encoding is self-checking: every fixture is loaded
through onnxruntime.InferenceSession inside the tests, so a byte this
module gets wrong fails the suite loudly. Field numbers are from
onnx/onnx.proto (ir_version 8, default domain, opset 13).
"""

from pathlib import Path

# TensorProto.DataType values (onnx/onnx.proto).
FLOAT, FLOAT16, DOUBLE = 1, 10, 11


def _varint(n: int) -> bytes:
    """Unsigned LEB128 — protobuf's varint (all our values are small +ve)."""
    out = bytearray()
    while True:
        n, low = n >> 7, n & 0x7F
        out.append(low | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _int(field: int, value: int) -> bytes:
    """A varint-typed field (wire type 0)."""
    return _varint(field << 3) + _varint(value)


def _blob(field: int, payload: bytes) -> bytes:
    """A length-delimited field (wire type 2): submessage, string, bytes."""
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _string(field: int, text: str) -> bytes:
    return _blob(field, text.encode())


def _value_info(name: str, elem_type: int, dims: tuple[int, ...]) -> bytes:
    """ValueInfoProto: name(1) + TypeProto(2){tensor_type(1){elem_type(1),
    shape(2){dim(1){dim_value(1)}...}}} — all dims static ints."""
    shape = b"".join(_blob(1, _int(1, d)) for d in dims)  # TensorShapeProto
    tensor = _int(1, elem_type) + _blob(2, shape)  # TypeProto.Tensor
    return _string(1, name) + _blob(2, _blob(1, tensor))


def flatten_model(elem_type: int, input_shape: tuple[int, ...]) -> bytes:
    """A ModelProto: Flatten(x) -> y, x of `input_shape`, y (batch, rest)."""
    n = 1
    for d in input_shape[1:]:
        n *= d
    node = (
        _string(1, "x") + _string(2, "y")  # input, output
        + _string(3, "flatten") + _string(4, "Flatten")  # name, op_type
    )
    graph = (
        _blob(1, node)
        + _string(2, "tiny")
        + _blob(11, _value_info("x", elem_type, input_shape))
        + _blob(12, _value_info("y", elem_type, (input_shape[0], n)))
    )
    opset = _int(2, 13)  # OperatorSetIdProto: default domain, version 13
    return _int(1, 8) + _blob(7, graph) + _blob(8, opset)  # ir_version 8


def write_model(directory: Path, filename: str, elem_type: int,
                input_shape: tuple[int, ...]) -> Path:
    """Serialize a flatten_model to `directory/filename` and return the path."""
    path = Path(directory) / filename
    path.write_bytes(flatten_model(elem_type, input_shape))
    return path

# Serialization documentation

## Libraries
We use ONNX and Protobuf to Serialize our data in C#.
- [Protobuf C# doc](https://developers.google.com/protocol-buffers/docs/reference/csharp-generated)
- [ONNX repo](https://github.com/onnx/onnx)

## Contributing
If you plan to add a serialization capacity to a function, check first that this function has a well defined spec [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and implements it accordingly. If not, just implement the new serialization process and mark it with a comment as "experimental".

### Good practice
- Since the default value of the I filed is 0, serializing will optimize and remove the field, creating an error with the officiel checker. 
  - Good practice: default value -> remove the node directly

- ONNX enforce the orders of tensor dimensions. This is why some operations have some "trans\*" fields indicating if a tensor needs to be transposed or not before being applied. 
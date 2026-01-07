# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter(
    data: onnxscript.FLOAT,
    position_ids: onnxscript.INT32,
    updates: onnxscript.FLOAT,
    is_keys: onnxscript.BOOL,  # True → keys, False → values
) -> onnxscript.FLOAT:
    # ---------------- VALUES BRANCH ----------------
    def values_branch():
        batch_size = ops.Gather(ops.Shape(data), [0])
        num_heads = ops.Gather(ops.Shape(data), [1])
        seq_len = ops.Gather(ops.Shape(position_ids), [1])

        zero = ops.Constant(value_ints=[0])
        one = ops.Constant(value_ints=[1])

        exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)

        batch_idx = ops.Expand(
            ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]),
            exp_shape,
        )
        head_idx = ops.Expand(
            ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]),
            exp_shape,
        )
        ctx_idx = ops.Expand(
            ops.Unsqueeze(position_ids, [1, 3]),
            exp_shape,
        )

        indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)
        return ops.ScatterND(data, indices, updates)

    # ---------------- KEYS BRANCH ----------------
    def keys_branch():
        shape = ops.Shape(data)
        batch_size = ops.Gather(shape, [0])
        num_heads = ops.Gather(shape, [1])
        head_dim = ops.Gather(shape, [2])
        seq_len = ops.Gather(ops.Shape(position_ids), [1])

        zero = ops.Constant(value_ints=[0])
        one = ops.Constant(value_ints=[1])

        exp_shape = ops.Concat(batch_size, num_heads, head_dim, seq_len, axis=0)

        batch_idx = ops.Expand(
            ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]),
            exp_shape,
        )
        head_idx = ops.Expand(
            ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]),
            exp_shape,
        )
        d_idx = ops.Expand(
            ops.Unsqueeze(ops.Range(zero, head_dim, one), [0, 1, 3]),
            exp_shape,
        )
        ctx_idx = ops.Expand(
            ops.Unsqueeze(position_ids, [1, 2]),
            exp_shape,
        )

        indices = ops.Concat(batch_idx, head_idx, d_idx, ctx_idx, axis=3)

        # Torch: updates.transpose(2, 3)
        updates_t = ops.Transpose(updates, perm=[0, 1, 3, 2])

        return ops.ScatterND(data, indices, updates_t)

    # ---------------- IF ----------------
    return ops.If(is_keys, keys_branch, values_branch)


class CtxScatterFunc(torch.autograd.Function):
    """
    Function to scatter the current key values into KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor, identifier: str):
        if identifier == "keys":
            batch_idx = torch.arange(data.shape[0]).view(-1, 1, 1, 1)
            head_idx = torch.arange(data.shape[1]).view(1, -1, 1, 1)
            d_idx = torch.arange(data.shape[2]).view(1, 1, -1, 1)
            ctx_idx = position_ids.view(data.shape[0], 1, 1, -1)
            data[batch_idx, head_idx, d_idx, ctx_idx] = updates.transpose(2, 3).contiguous()

        elif identifier == "values":
            batch_idx = torch.arange(data.shape[0]).view(-1, 1, 1)
            head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
            ctx_idx = position_ids.unsqueeze(1)
            data[batch_idx, head_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        data: torch.Value,
        position_ids: torch.Value,
        updates: torch.Value,
        identifier: str,
    ) -> torch.Value:
        # Build a BOOL tensor constant: True for "keys", False for "values"
        is_keys_const = g.op(
            "Constant",
            value_t=torch.tensor(identifier == "keys", dtype=torch.bool),
        )

        return g.onnxscript_op(
            CtxScatter,
            data,
            position_ids,
            updates,
            is_keys_const,
        ).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter3D(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = torch.arange(data.shape[0]).view(-1, 1)
        ctx_idx = position_ids
        data[batch_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3D, data, position_ids, updates).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGather3D(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Expand(ctx_indices, ops.Slice(ops.Shape(data), starts=[0], ends=[2], axes=[0]))
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=1)


class CtxGatherFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1)
        return data[batch_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather3D, data, ctx_indices).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGather(
    data: onnxscript.FLOAT,
    ctx_indices: onnxscript.INT32,
    comp_ctx_len: onnxscript.INT64,  # <-- Use INT64 for shape
    is_keys: onnxscript.BOOL,
) -> onnxscript.FLOAT:
    # -------- values branch --------
    def values_branch():
        # Create a shape tensor based on comp_ctx_len
        shape_tensor = ops.Concat(ops.Shape(data)[:2], ops.Reshape(comp_ctx_len, [1]), axis=0)

        # Directly use the shape tensor without validation
        ctx_indices = ops.Expand(ctx_indices, shape_tensor)
        ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
        return ops.GatherND(data, ctx_indices, batch_dims=2)

    # -------- keys branch --------
    def keys_branch():
        shape_tensor = ops.Shape(data)[:3]

        # Build index grids
        b_idx = ops.Expand(ops.Reshape(ops.Range(0, ops.Gather(shape_tensor, 0), 1), [-1, 1, 1]), shape_tensor)
        h_idx = ops.Expand(ops.Reshape(ops.Range(0, ops.Gather(shape_tensor, 1), 1), [1, -1, 1]), shape_tensor)
        s_idx = ops.Expand(ops.Reshape(ops.Range(0, ops.Gather(shape_tensor, 2), 1), [1, 1, -1]), shape_tensor)

        # Combine indices
        indices = ops.Concat(
            ops.Unsqueeze(b_idx, [-1]), ops.Unsqueeze(h_idx, [-1]), ops.Unsqueeze(s_idx, [-1]), axis=-1
        )

        return ops.GatherND(data, indices, batch_dims=0)

    # -------- ONNX If --------
    return ops.If(is_keys, values_branch, keys_branch)


class CtxGatherFunc(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int, identifier: str):
        if identifier == "keys":
            batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
            head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
            seq_indices = torch.arange(data.shape[2]).view(1, 1, -1)
            return data[batch_indices, head_indices, seq_indices]

        elif identifier == "values":
            batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
            head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
            ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
            return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        data: torch.Value,
        ctx_indices: torch.Value,
        comp_ctx_len,  # may be Python int OR torch._C.Value
        identifier: str,
    ) -> torch.Value:
        if not isinstance(ctx_indices, torch.Value):
            ctx_indices = g.op("Constant", value_t=torch.tensor(ctx_indices, dtype=torch.int64))

        is_keys_const = g.op(
            "Constant",
            value_t=torch.tensor(identifier == "values", dtype=torch.bool),
        )

        return g.onnxscript_op(
            CtxGather,
            data,
            ctx_indices,
            comp_ctx_len,
            is_keys_const,
        ).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGatherBlockedKV(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=2)


class CtxGatherFuncBlockedKV(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherBlockedKV, data, ctx_indices).setTypeAs(data)

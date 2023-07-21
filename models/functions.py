import torch


class _NormalizeGradients(torch.autograd.Function):
    """
    Used to normalize gradients at any intermediate layer (especially in an MTL setting)
    Usage:
    if your code looks like:
        for output_name, mod in self.output_mlps.items():
            output[output_name] = mod(shared_representation)
    you may replace it with this simple hook to normalize gradients flowing into "shared_representation"
        for output_name, mod in self.output_mlps.items():
            output[output_name] = mod(normalize_gradients(shared_representation))
    """
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output / (torch.norm(grad_output) + 1e-6)
        return grad_input


normalize_gradients = _NormalizeGradients.apply

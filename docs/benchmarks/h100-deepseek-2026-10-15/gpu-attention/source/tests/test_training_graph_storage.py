#!/usr/bin/env python3
"""CPU reproduction of gradient aliasing when a replay buffer is reused."""
import json
import torch


class ReplayGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, storage, value):
        ctx.storage, ctx.value = storage, value
        return parameter.sum()

    @staticmethod
    def backward(ctx, incoming):
        ctx.storage.fill_(ctx.value)
        return ctx.storage.detach(), None, None


def main():
    records = []
    for reserved in (False, True):
        parameter = torch.nn.Parameter(torch.zeros(8))
        storage = torch.empty_like(parameter)
        if reserved:
            parameter.grad = torch.zeros_like(parameter)
        ReplayGradient.apply(parameter, storage, 1.).backward()
        aliases = parameter.grad.data_ptr() == storage.data_ptr()
        ReplayGradient.apply(parameter, storage, 3.).backward()
        expected = 4. if reserved else 6.
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, expected), rtol=0, atol=0)
        assert aliases != reserved
        records.append(dict(reserved=reserved, gradient_aliases_replay=aliases,
                            accumulated_gradient=parameter.grad[0].item(), expected_sum=4.))
    print(json.dumps(dict(status='passed',cases=records)),flush=True)


if __name__ == '__main__':
    main()

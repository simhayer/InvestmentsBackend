import asyncio
import unittest

from pydantic import BaseModel

from agent.state_models import ToolBudget
from agent.tool_executor import ToolCallSpec, ToolExecutor
from services.ai.chat_agent.tools import ToolContext


class BoomInput(BaseModel):
    value: int = 0


async def boom_tool(_args: BoomInput, _ctx: ToolContext):
    raise RuntimeError("boom")


class ToolEnvelopeWrapperTests(unittest.TestCase):
    def test_exception_wrapped(self):
        executor = ToolExecutor(
            tool_registry={
                "boom_tool": type(
                    "Spec",
                    (),
                    {
                        "input_model": BoomInput,
                        "run": boom_tool,
                    },
                )()
            }
        )
        ctx = ToolContext(
            db=None,
            finnhub=None,
            user_id=None,
            user_currency="USD",
            message="",
            symbols=[],
            history=[],
            holdings_snapshot=None,
        )
        budgets = {"boom_tool": ToolBudget(max_calls=1, timeout_s=0.1)}
        calls = [ToolCallSpec(name="boom_tool", arguments={"value": 1}, data_type="test")]

        async def run():
            return await executor.execute(calls, ctx, budgets, global_timeout_s=1.0)

        results, _ = asyncio.run(run())
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].error.type, "tool_error")


if __name__ == "__main__":
    unittest.main()

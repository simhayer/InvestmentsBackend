import asyncio
import unittest

from pydantic import BaseModel

from agent.state_models import ToolBudget
from agent.tool_executor import ToolCallSpec, ToolExecutor
from services.ai.chat_agent.tools import ToolContext


class SleepInput(BaseModel):
    duration: float = 0.0


async def sleep_tool(args: SleepInput, _ctx: ToolContext):
    await asyncio.sleep(args.duration)
    return {"slept": args.duration}


class ToolCapsAndTimeoutsTests(unittest.TestCase):
    def test_cap_exceeded(self):
        executor = ToolExecutor(
            tool_registry={
                "sleep_tool": type(
                    "Spec",
                    (),
                    {
                        "input_model": SleepInput,
                        "run": sleep_tool,
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
        budgets = {"sleep_tool": ToolBudget(max_calls=1, timeout_s=0.1)}
        calls = [
            ToolCallSpec(name="sleep_tool", arguments={"duration": 0.0}, data_type="test"),
            ToolCallSpec(name="sleep_tool", arguments={"duration": 0.0}, data_type="test"),
        ]

        async def run():
            return await executor.execute(calls, ctx, budgets, global_timeout_s=1.0)

        results, _ = asyncio.run(run())
        error_types = [res.error.type for res in results if res.error]
        self.assertIn("cap_exceeded", error_types)

    def test_timeout(self):
        executor = ToolExecutor(
            tool_registry={
                "sleep_tool": type(
                    "Spec",
                    (),
                    {
                        "input_model": SleepInput,
                        "run": sleep_tool,
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
        budgets = {"sleep_tool": ToolBudget(max_calls=1, timeout_s=0.01)}
        calls = [ToolCallSpec(name="sleep_tool", arguments={"duration": 0.05}, data_type="test")]

        async def run():
            return await executor.execute(calls, ctx, budgets, global_timeout_s=1.0)

        results, _ = asyncio.run(run())
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].error.type, "timeout")


if __name__ == "__main__":
    unittest.main()

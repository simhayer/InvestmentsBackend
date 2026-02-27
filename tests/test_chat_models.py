import unittest

from pydantic import ValidationError

from services.ai.chat.chat_models import ChatMessage, format_sse


class TestChatModels(unittest.TestCase):
    def test_chat_message_trims_content(self):
        msg = ChatMessage(role="user", content="  hello  ")
        self.assertEqual(msg.content, "hello")

    def test_chat_message_rejects_empty_content(self):
        with self.assertRaises(ValidationError):
            ChatMessage(role="user", content="  ")

    def test_format_sse_shape(self):
        out = format_sse("token", {"text": "hi"})
        self.assertTrue(out.startswith("event: token\n"))
        self.assertIn("data: ", out)
        self.assertTrue(out.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()

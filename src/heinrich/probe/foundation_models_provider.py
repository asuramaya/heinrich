"""Provider for Apple's on-device Foundation Models (macOS/iOS 26+).

Bridges Heinrich's Python probe tooling to Apple's LanguageModelSession
via a Swift subprocess that speaks JSON over stdin/stdout.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Sequence

from ..core.signal import Signal


# Swift helper source — compiled once and cached.
_SWIFT_HELPER = textwrap.dedent(r"""
    import FoundationModels
    import Foundation

    struct Request: Decodable {
        let id: String
        let instructions: String?
        let messages: [Message]

        struct Message: Decodable {
            let role: String
            let content: String
        }
    }

    struct Response: Encodable {
        let id: String
        let text: String
        let error: String?
        let latency: Double
    }

    let decoder = JSONDecoder()
    let encoder = JSONEncoder()

    while let line = readLine(strippingNewline: true) {
        guard let data = line.data(using: .utf8) else { continue }
        guard let request = try? decoder.decode(Request.self, from: data) else {
            let errResp = try! encoder.encode(Response(id: "?", text: "", error: "parse_error", latency: 0))
            print(String(data: errResp, encoding: .utf8)!)
            fflush(stdout)
            continue
        }

        let instructions = request.instructions ?? "You are a helpful assistant."
        let session = LanguageModelSession(instructions: instructions)

        let userContent = request.messages.last?.content ?? ""
        let start = CFAbsoluteTimeGetCurrent()
        var responseText = ""
        var errorText: String? = nil

        do {
            let result = try await session.respond(to: userContent)
            responseText = result.content
        } catch {
            errorText = String(describing: error)
            if "\(error)".contains("guardrailViolation") {
                errorText = "guardrail_violation"
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let resp = Response(id: request.id, text: responseText, error: errorText, latency: elapsed)
        let respData = try! encoder.encode(resp)
        print(String(data: respData, encoding: .utf8)!)
        fflush(stdout)
    }
""")


class FoundationModelsProvider:
    """Heinrich Provider backed by Apple's on-device Foundation Models.

    Launches a persistent Swift subprocess that accepts JSON requests
    on stdin and returns JSON responses on stdout. Session state is
    per-request (stateless from Heinrich's perspective).
    """

    def __init__(self, *, instructions: str | None = None) -> None:
        self._instructions = instructions
        self._process: subprocess.Popen | None = None
        self._swift_binary: Path | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "provider_type": "foundation_models",
            "platform": "apple_on_device",
            "instructions": self._instructions,
        }

    def _ensure_running(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process

        if self._swift_binary is None or not self._swift_binary.exists():
            self._swift_binary = self._compile_helper()

        self._process = subprocess.Popen(
            [str(self._swift_binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._process

    def _compile_helper(self) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="heinrich_fm_"))
        src = tmp_dir / "fm_bridge.swift"
        binary = tmp_dir / "fm_bridge"
        src.write_text(_SWIFT_HELPER)

        result = subprocess.run(
            ["swiftc", "-O", "-o", str(binary), str(src)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile Foundation Models bridge:\n{result.stderr}")
        return binary

    def _send_request(self, request_id: str, content: str, instructions: str | None = None) -> dict[str, Any]:
        proc = self._ensure_running()

        request = {
            "id": request_id,
            "instructions": instructions or self._instructions,
            "messages": [{"role": "user", "content": content}],
        }

        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Foundation Models bridge returned empty response")
        return json.loads(line)

    def chat_completions(
        self, cases: Sequence[dict[str, Any]], *, model: str = "apple_fm"
    ) -> list[dict[str, Any]]:
        results = []
        for case in cases:
            cid = case.get("custom_id", "")
            messages = case.get("messages", [])
            content = messages[-1]["content"] if messages else ""

            # Extract system/instructions from messages if present
            instructions = self._instructions
            for msg in messages:
                if msg.get("role") == "system":
                    instructions = msg["content"]
                    break

            resp = self._send_request(cid, content, instructions)
            results.append({
                "custom_id": cid,
                "text": resp.get("text", ""),
                "error": resp.get("error"),
                "latency": resp.get("latency", 0),
            })
        return results

    def activations(
        self, cases: Sequence[dict[str, Any]], *, model: str = "apple_fm"
    ) -> list[dict[str, Any]]:
        # Black box — no activation access
        return [{"custom_id": case.get("custom_id", ""), "activations": {}} for case in cases]

    def close(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            self._process.terminate()
            self._process = None

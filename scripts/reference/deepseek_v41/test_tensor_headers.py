"""Offline tests: fixture bytes are headers, never tensor values."""

import copy
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import urllib.request
import urllib.response

import tensor_headers as gate


class Response(io.BytesIO):
    def __init__(self, body=b"12345678", status=206, **headers):
        super().__init__(body)
        self.status = status
        self.headers = {
            "Content-Length": "8",
            "Content-Range": "bytes 0-7/100",
            **headers,
        }
        self.read_calls = []

    def read(self, count=-1):
        self.read_calls.append(count)
        return super().read(count)


class HeaderTests(unittest.TestCase):
    def test_real_redirect_handler_never_drains_bodies(self):
        bodies, requests = [], []

        class Transport(urllib.request.HTTPSHandler):
            def https_open(self, request):
                requests.append(request)
                body = Response()
                bodies.append(body)
                redirect = len(requests) == 1
                headers = (
                    {
                        "Location": "https://cdn.example/header",
                        "Content-Length": str(2**60),
                    }
                    if redirect
                    else {"Content-Length": "8", "Content-Range": "bytes 0-7/100"}
                )
                response = urllib.response.addinfourl(
                    body, headers, request.full_url, 302 if redirect else 206
                )
                response.msg = "fixture"
                return response

        opener = urllib.request.build_opener(gate.BoundedRedirects(), Transport())
        reader = gate.Reader(opener.open)
        self.assertEqual(reader.fetch("shard", 0, 8), (b"12345678", 100))
        self.assertEqual(reader.used, 8)
        self.assertEqual(bodies[0].read_calls, [])
        self.assertTrue(bodies[0].closed)
        self.assertEqual(bodies[1].read_calls, [8])
        self.assertTrue(all(r.get_header("Range") == "bytes=0-7" for r in requests))
        self.assertEqual(requests[1].get_header("Host"), "cdn.example")

    def test_redirect_hops_and_scheme_are_bounded(self):
        for location, error, expected_requests in [
            (
                "https://cdn.example/loop",
                "too many",
                gate.BoundedRedirects.MAX_HOPS + 1,
            ),
            ("http://cdn.example/header", "non-HTTPS", 1),
            ("file:///not-a-network-resource", "non-HTTPS", 1),
            (None, "missing Location", 1),
        ]:
            bodies = []

            class Transport(urllib.request.HTTPSHandler):
                def https_open(self, request):
                    body = Response()
                    bodies.append(body)
                    response = urllib.response.addinfourl(
                        body, {"Location": location}, request.full_url, 302
                    )
                    response.msg = "fixture"
                    return response

            opener = urllib.request.build_opener(gate.BoundedRedirects(), Transport())
            with (
                self.subTest(location=location),
                self.assertRaisesRegex(ValueError, error),
            ):
                gate.Reader(opener.open).fetch("shard", 0, 8)
            self.assertEqual(len(bodies), expected_requests)
            self.assertTrue(
                all(body.closed and body.read_calls == [] for body in bodies)
            )

    def test_recapture_cannot_replace_trust_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            original = {"sha256": "original"}
            gate.check_existing_manifest(path, original)
            path.write_text(json.dumps(original))
            gate.check_existing_manifest(path, original)
            with self.assertRaisesRegex(ValueError, "differs from the pinned"):
                gate.check_existing_manifest(path, {"sha256": "changed"})
            self.assertEqual(json.loads(path.read_text()), original)

    def test_real_fixtures_are_metadata_not_values(self):
        manifest = gate.verify()
        self.assertEqual(len(manifest["tensors"]), 18)
        for entry in manifest["shards"].values():
            self.assertEqual(
                (gate.ROOT / entry["fixture"]).stat().st_size,
                entry["header_length"] + 8,
            )
            self.assertGreater(
                entry["source_content_length"], entry["header_length"] + 8
            )
        for key, value in manifest["tensors"].items():
            if ".experts." in key and key.endswith("weight"):
                self.assertEqual(value["descriptor"]["dtype"], "I8")
                self.assertNotIn("values", value)

    def test_strict_ranges_reject_before_read(self):
        cases = [
            {"status": 200},
            {"Content-Range": "bytes 0-7/*"},
            {"Content-Range": "bytes 1-8/100"},
            {"Content-Range": "bytes 0-7/7"},
            {"Content-Range": "bytes 0-7/100 trailing"},
            {"Content-Length": "100000000000"},
            {"Content-Length": ""},
            {"Content-Encoding": "gzip"},
        ]
        for case in cases:
            with self.subTest(case=case):
                response = Response(**case)
                reader = gate.Reader(lambda *a, **kw: response)
                with self.assertRaises(ValueError):
                    reader.fetch("shard", 0, 8)
                self.assertEqual(response.read_calls, [])

    def test_exact_bounded_read_and_request(self):
        response = Response()

        def open_response(request, timeout):
            self.assertEqual(request.get_header("Range"), "bytes=0-7")
            self.assertIn(gate.REVISION, request.full_url)
            return response

        reader = gate.Reader(open_response)
        self.assertEqual(reader.fetch("shard", 0, 8), (b"12345678", 100))
        self.assertEqual(response.read_calls, [8])
        self.assertEqual(reader.used, 8)

    def test_truncation_and_changed_total(self):
        for response, total in [(Response(body=b"short"), None), (Response(), 101)]:
            with self.assertRaises(ValueError):
                gate.Reader(lambda *a, **kw: response).fetch("shard", 0, 8, total)

    def test_bounds_without_network(self):
        def forbidden(*a, **kw):
            self.fail("network called for invalid request")

        reader = gate.Reader(forbidden)
        for length in (0, gate.MAX_HEADER + 1):
            with self.assertRaises(ValueError):
                reader.fetch("shard", 8, length)
        reader.used = gate.MAX_TOTAL - 7
        with self.assertRaises(ValueError):
            reader.fetch("shard", 0, 8)
        for length in (0, 7, gate.MAX_HEADER + 8, 2**64 - 1):
            with self.assertRaises(ValueError):
                gate.header_length(struct.pack("<Q", length))

    def test_index_bound_before_read(self):
        response = Response(status=200, **{"Content-Length": str(gate.MAX_INDEX + 1)})
        with self.assertRaises(ValueError):
            gate.Reader(lambda *a, **kw: response).fetch("model.safetensors.index.json")
        self.assertEqual(response.read_calls, [])

    def test_invalid_descriptors(self):
        valid = {
            "a": {"dtype": "I8", "shape": [8], "data_offsets": [0, 8]},
            "b": {"dtype": "I8", "shape": [8], "data_offsets": [8, 16]},
        }

        def validate(obj):
            raw = json.dumps(obj).encode()
            raw += b" " * (-len(raw) % 8)
            return gate.descriptors(raw, len(raw) + 8 + 16)

        validate(valid)
        for field, value in (
            ("shape", [7]),
            ("shape", [-8]),
            ("dtype", "F32"),
            ("data_offsets", [4, 12]),
            ("data_offsets", [9, 17]),
        ):
            obj = copy.deepcopy(valid)
            obj["b"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate(obj)

    def test_manifest_descriptor_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(gate.ROOT / "fixtures/headers", root / "fixtures/headers")
            for name in gate.SOURCE_HASHES:
                target = root / "fixtures" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(gate.ROOT / "fixtures" / name, target)
            manifest = json.loads((gate.ROOT / "tensor-headers.json").read_text())
            key = "layers.0.ffn.experts.0.w1.weight"
            manifest["tensors"][key]["descriptor"]["dtype"] = "U8"
            (root / "tensor-headers.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mismatched selected descriptors"):
                gate.verify(root)
            shutil.copyfile(
                gate.ROOT / "tensor-headers.json", root / "tensor-headers.json"
            )
            entry = next(iter(manifest["shards"].values()))
            fixture = root / entry["fixture"]
            blob = fixture.read_bytes()
            fixture.write_bytes(blob + b"not a tensor value fixture")
            with self.assertRaisesRegex(ValueError, "fixture contains payload"):
                gate.verify(root)
            fixture.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "header digest mismatch"):
                gate.verify(root)

    def test_geometry_rejects_incorrect_scale(self):
        manifest = gate.verify()
        key = "layers.0.ffn.experts.0.w1.weight"
        weight = manifest["tensors"][key]["descriptor"]
        scale = copy.deepcopy(
            manifest["tensors"][key.replace("weight", "scale")]["descriptor"]
        )
        scale["shape"] = [72, 160]
        with self.assertRaises(ValueError):
            gate.geometry(key, weight, scale)


if __name__ == "__main__":
    unittest.main()

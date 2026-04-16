import pytest

from ai_service.services.booking_api import BookingAPI


class _StubAsyncResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _StubAsyncClient:
    def __init__(self):
        self.last_url = None

    async def get(self, url, *args, **kwargs):
        self.last_url = url
        # Return a stubbed response with deterministic JSON
        return _StubAsyncResponse({"ok": True, "url": url})


@pytest.mark.asyncio
async def test_fetch_business_information_calls_expected_endpoint_and_returns_json():
    api = BookingAPI()
    stub_client = _StubAsyncClient()
    api._client = stub_client  # inject stub client

    data = await api.fetch_business_information()
    print("Data:: ", data)
    assert isinstance(data, dict)
    assert data.get("ok") is True
    # Ensure correct URL was requested
    expected_prefix = "https://prod.snapsbooking.com/api/salon-bookings/pedinnail-by-mynyjona/"
    
    assert stub_client.last_url == expected_prefix
    assert data.get("url") == expected_prefix


def test_get_booking_information_currently_unimplemented_returns_none():
    api = BookingAPI()
    # The second definition of get_booking_information overrides the first and returns None
    result = api.get_booking_information("ABC123")
    assert result is None



import httpx


class BookingAPI:
    """Booking API for the receptionist."""
    _api_url: str
    _client: httpx.AsyncClient

    def __init__(self):
        """Initialize the booking API."""
        self._api_url = f"http://3.17.232.108:8002/api/salon-bookings"
        self._client = httpx.AsyncClient()
        self._salon_id = 1

    async def fetch_business_information(self, info_type: str = "general"):
        """Get business information."""
        url = f"{self._api_url}/pedinnail-by-mynyjona/"
        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "e53a42f88611579206a28c8a183bc5d400b21875bce3b0c31e26d96dd98219da",
            "x-timestamp": "1759172432"
        }
        response = await self._client.get(url, headers=headers)
        data = response.json()
        print("Business information Data:: ", data.get("data"))
        return data.get("data")

    async def fetch_business_services(self, booking_id: str = "123") -> str:
        """Get business services."""
        url = f"{self._api_url}/services/?salon_id=1"
        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
            "x-timestamp": "1759177322"
        }
        response = await self._client.get(url, headers=headers)
        data = response.json()
        print("Business services Data:: ", data.get("data"))
        return data.get("data")

    async def check_availability(self, booking__selected_date: str, service_duration: str, service_ids: list[int]) -> str:
        """Check availability."""

        url = f"{self._api_url}/availability-slots/"
        params = {
            "booking__selected_date": booking__selected_date,
            "service_duration": service_duration,
            "salon_id": self._salon_id,
        }
        if service_ids:
            params["service_ids[]"] = service_ids
        print("Params:: ", params)
        print("URL:: ", url)

        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "5a49d5928896bfab4508aae1c915e50b842e49799947d8306a1b33ac463f7985",
            "x-timestamp": "1759178298"
        }
        response = await self._client.get(url, headers=headers, params=params)
        data = response.json()
        print("Availability Data:: ", data.get("data"))
        return data.get("data")

    async def search_services(self, service_name: str) -> str:
        """Search services."""
        url = f"{self._api_url}/services/"
        params = {
            "service_name": service_name,
            "salon_id": self._salon_id
        }
        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
            "x-timestamp": "1759177322"
        }
        response = await self._client.get(url, headers=headers, params=params)
        data = response.json()
        print("Services Data:: ", data.get("data"))

    async def book_appointment(self, appointment_data: dict) -> str:
        """Book appointment."""

        try:
            url = f"{self._api_url}/"

            print("booked appointment data:: ", appointment_data)
            headers = {
                "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
                "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
                "x-timestamp": "1759177322"
            }
            response = await self._client.post(url, headers=headers, json=appointment_data)
            data = response.json()
            print("Book appointment Data:: ", data.get("data"))
            return data.get("data")

        except Exception as e:
            print("Error booking appointment:: ", str(e))
            return None

    async def fetch_customer_information(self, customer_phone: str) -> str:
        """Fetch customer information."""
        try:
            url = f"{self._api_url}/salon-customer/"
            params = {
                "phone_number": customer_phone,
                "salon_id": self._salon_id
            }
            headers = {
                "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
                "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
                "x-timestamp": "1759177322"
            }
            response = await self._client.get(url, headers=headers, params=params)
            data = response.json()
            print("Customer information Data:: ", data.get("data"))
            return data.get("data")
        except Exception as e:
            print("Error fetching customer information:: ", str(e))
            return None

    async def create_customer(self, full_name: str, phone: str) -> str:
        """Create customer."""
        url = f"{self._api_url}/register-salon-customer/"
        body = {
            "phone_number": phone,
            "full_name": full_name,
            "salon_id": self._salon_id
        }
        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
            "x-timestamp": "1759177322"
        }
        response = await self._client.post(url, headers=headers, json=body)
        data = response.json()
        print("Customer created Data:: ", data.get("data"))
        return data.get("data")

    async def find_my_appointments(self, phone_number: str, date: str):
        """Get next appointments."""
        url = f"{self._api_url}/find-my-appointments/"
        params = {
            "phone_number": phone_number,
            "salon_id": self._salon_id
        }
        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
            "x-timestamp": "1759177322"
        }
        response = await self._client.get(url, headers=headers, params=params)
        data = response.json()
        return data.get("data")

    async def cancel_appointment(self, appointment_id: int, phone_number: str):
        """Cancel appointment."""
        url = f"{self._api_url}/cancel-my-appointment/"
        body = {
            "booking_id": appointment_id,
            "salon_id": self._salon_id,
            "phone_number": phone_number
        }
        headers = {
            "x-api-key": "BPgAVDVmpU3xjtnpSTQx_BNJ6KWu",
            "x-signature": "328f864442b52357916ad899757cc8df38aac0fc089fec9e20b6fc1bb283ace2",
            "x-timestamp": "1759177322"
        }
        response = await self._client.patch(url, headers=headers, json=body)
        data = response.json()
        return data
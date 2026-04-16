"""Health and test routes for the AI Receptionist application."""

from fastapi import APIRouter
from datetime import datetime
from ai_service.config import settings
from ai_service.services.booking_api import BookingAPI
from receptionist.models import SystemLog
import json

# Create router
router = APIRouter()

# get business information
@router.get("/business-information")
async def get_business_information():
    """Get business information."""
   
    business_services_sample = {
        "name": "SnapsBooking Salon",
        "phone": "(555) 123-4567",
        "email": "info@snapsbooking.com",
        "website": "www.snapsbooking.com",
        "address": "123 Salon Lane, Creative District, City, State 12345",
        "established": "2020"
    }
    print("Business services sample:: ", json.dumps(business_services_sample))
    return business_services_sample
    # return await BookingAPI().fetch_business_information()
  
# get business services
@router.get("/business-services")
async def get_business_services():
    """Get business services."""
    return await BookingAPI().fetch_business_services()
  

# check availability
@router.get("/check-availability")
async def check_availability(booking__selected_date: str, service_duration: str, service_ids: str = ""):
    """Check availability."""
    print("Booking selected date:: ", booking__selected_date)
    return await BookingAPI().check_availability(booking__selected_date, service_duration, service_ids)


# get next appointments
@router.get("/next-appointments")
async def get_next_appointments(phone_number: str):
    """Get next appointments."""
    return await BookingAPI().get_next_appointments(phone_number)
"""Business booking service layer that handles all booking-related operations."""
from typing import Dict, Any, List, Optional
from business.models import Business, OperatingHours, BusinessSettings
from service.models import Service, ServiceCategory
from client.models import Client
from appointment.models import Appointment, AppointmentService, AppointmentStatusType, BookingSourceType
from staff.models import Staff, StaffService, StaffWorkingHours, StaffOffDay
from django.utils import timezone
from django.db import transaction, models
from asgiref.sync import sync_to_async
from datetime import datetime, timedelta, time
import json
from appointment.services import BusinessBookingService as AppointmentBusinessBookingService
from appointment.serializers import AppointmentDetailSerializer
from business.serializers import BusinessSerializer
from service.serializers import ServiceSerializer
from client.serializers import ClientSerializer
from staff.serializers import StaffSerializer
from logging import getLogger
import re
logger = getLogger(__name__)


class BusinessBookingService:
    """Service layer for handling business booking operations."""

    def __init__(self, business_id: int):
        """
        Initialize the business booking service.

        Args:
            business_id: ID of the business this service operates for
        """
        self.business_id = business_id

    @sync_to_async
    def _get_business(self):
        """Get the business instance."""
        return Business.objects.get(id=self.business_id)

    @sync_to_async
    def _get_business_info_sync(self, info_type: str = "general") -> Dict[str, Any]:
        """
        Get business information synchronously.
        
        Args:
            info_type: Type of information to retrieve (general, hours, contact, etc.)
            
        Returns:
            Dictionary containing business information
        """
        business = Business.objects.select_related('business_type').get(id=self.business_id)
        operating_hours = OperatingHours.objects.filter(business=business).order_by('day_of_week')
        
        # Serialize operating hours
        hours_data = []
        for oh in operating_hours:
            day_name = dict(OperatingHours.DAY_CHOICES)[oh.day_of_week]
            hours_data.append({
                'day': day_name,
                'day_of_week': oh.day_of_week,
                'is_open': oh.is_open,
                'open_time': oh.open_time.strftime('%H:%M') if oh.open_time else None,
                'close_time': oh.close_time.strftime('%H:%M') if oh.close_time else None,
                'is_break_time': oh.is_break_time,
                'break_start_time': oh.break_start_time.strftime('%H:%M') if oh.break_start_time else None,
                'break_end_time': oh.break_end_time.strftime('%H:%M') if oh.break_end_time else None,
            })
        
        return {
            'id': str(business.id),
            'name': business.name,
            'business_type': business.business_type.name if business.business_type else None,
            'phone_number': business.phone_number,
            'twilio_phone_number': business.twilio_phone_number,
            'email': business.email,
            'website': business.website,
            'address': business.address,
            'city': business.city,
            'state_province': business.state_province,
            'postal_code': business.postal_code,
            'country': business.country,
            'description': business.description,
            'status': business.status,
            'operating_hours': hours_data,
            'google_review_url': business.google_review_url,
        }

    async def get_business_information(self, info_type: str = "general") -> Dict[str, Any]:
        """
        Get business information.
        
        Args:
            info_type: Type of information to retrieve (general, hours, contact, etc.)
            
        Returns:
            Dictionary containing business information
        """
        return await self._get_business_info_sync(info_type)

    @sync_to_async
    def _get_services_sync(self) -> List[Dict[str, Any]]:
        """
        Get service information synchronously.
        
        Args:
            service_type: Type of service to retrieve
            
        Returns:
            List of services with their details
        """
        services = Service.objects.filter(
            business_id=self.business_id,
            is_active=True,
            is_online_booking=True
        ).select_related('category').order_by('sort_order', 'name')
        
        logger.info(f"Services:: {services}")
        
        return [self._serialize_service(service) for service in services]

    async def get_service_information(self) -> List[Dict[str, Any]]:
        """
        Get service information.
        
        Args:
            service_type: Type of service to retrieve
            
        Returns:
            List of services with their details
        """
        return await self._get_services_sync()

    async def check_availability(
        self,
        date: str,
        time: str = "any",
        service_ids: list[int] = None,
        service_duration: int = None,
        staff_id: int = None,
    ) -> Dict[str, Any]:
        logger.info(f"Checking availability: date={date}, time={time}, service_ids={service_ids}, service_duration={service_duration}, staff_id={staff_id}")

    
        if not service_ids:
            return {'available_slots': [], 'error': f'No services found matching: {service_ids}'}
        
        if not service_duration:
            return {'available_slots': [], 'error': f'No service duration found'}

        availability_data = await self._check_availability_sync(
            booking_date=date,
            service_duration=service_duration,
            service_ids=service_ids,
            staff_id=staff_id,
        )

        logger.info(f"Availability data: {availability_data}")
        logger.info(f"Found {availability_data.get('total_slots', 0)} available slots")
        return availability_data
    
    async def search_services_by_keywords(self, keywords: list[str]) -> list[Service]:
        """
        Search for services by keywords.
        """
        return await self._search_services_by_keywords_sync(keywords=keywords)

    @sync_to_async
    def _search_services_by_keywords_sync(self, keywords: list[str]) -> list[Service]:
        """
        Search for services by keywords.
        """
        query = models.Q()
        for keyword in keywords:
            query |= models.Q(name__icontains=keyword) | models.Q(description__icontains=keyword)

        services = Service.objects.filter(query)
        logger.info(f"========== Resolved Search services by keywords: {keywords} -> {services} ==========")
        return [self._serialize_service(service) for service in services]

    @sync_to_async
    def _resolve_services(self, service_keywords: list[str]) -> list[Service]:
        """
        Look up services by name/description and return their IDs and total duration.

        Args:
            service_keywords: Keywords to search for services by name or description (e.g. ["haircut", "hair styling", "hair coloring", "manicure", "pedicure", "facial", "massage", "etc"]).

        Returns:
            List of services
        """
        if not service_keywords:
            return []
        
        # Build Q objects for each keyword
        query = models.Q()
        for keyword in service_keywords:
            query |= models.Q(name__icontains=keyword) | models.Q(description__icontains=keyword)

        services = Service.objects.filter(
            business_id=self.business_id,
            is_active=True,
            is_online_booking=True,
        ).filter(query)
        
        logger.info(f"========== Resolved services: {services} ==========")


        return list(services)

    def _get_staff_name(self, staff_id: int) -> str:
        """
        Get staff name by ID.
        
        Args:
            staff_id: ID of the staff member
            
        Returns:
            Full name of the staff member
        """
        try:
            staff = Staff.objects.only('first_name', 'last_name').get(id=staff_id)
            return staff.get_full_name()
        except Staff.DoesNotExist:
            return "Unknown Staff"

    @sync_to_async
    def _check_availability_sync(
        self,
        booking_date: str,
        service_duration: int,
        service_ids: List[int],
        staff_id: int = None
    ) -> Dict[str, Any]:
        """
        Check availability using AppointmentBusinessBookingService.
        
        Args:
            booking_date: Date string in YYYY-MM-DD format
            service_duration: Total duration in minutes
            service_ids: List of service IDs requested
            staff_id: Staff ID to check availability for
        Returns:
            Dictionary with available slots
        """
        try:
            # Get business settings for interval configuration
            try:
                settings = BusinessSettings.objects.get(business_id=self.business_id)
                interval_minutes = settings.time_slot_interval
            except BusinessSettings.DoesNotExist:
                interval_minutes = 15
            
            # Use existing appointment booking service
            appointment_service = AppointmentBusinessBookingService(
                business_id=self.business_id,
                interval_minutes=interval_minutes
            )
            
            logger.info(f"Checking availability for staff: {staff_id}")
            # Get available time slots
            if staff_id:
                logger.info(f"Checking availability for staff: {staff_id}")
                time_slots = appointment_service.get_staff_time_slots(
                    staff_id=staff_id,
                    service_ids=service_ids,
                    appointment_date=booking_date,
                    service_duration=service_duration,
                )
            else:
                logger.info(f"Checking availability for all staff")
                time_slots = appointment_service.get_all_available_time_slots(
                    business_id=self.business_id,
                    service_ids=service_ids,
                    appointment_date=booking_date,
                    service_duration=service_duration
                )
            
            # Transform output format to match expected structure
            formatted_slots = []
            for slot in time_slots:
                formatted_slots.append({
                    'staff_id': slot['staff_id'],
                    'start_at': slot['start_time'].isoformat(),
                    'end_at': slot['end_time'].isoformat(),
                    'start_time': slot['start_time'].strftime('%H:%M'),
                    'end_time': slot['end_time'].strftime('%H:%M'),
                    'duration': service_duration
                })
            
            return {
                'available_slots': formatted_slots,
                'date': booking_date,
                'service_duration': service_duration,
                'total_slots': len(formatted_slots)
            }
            
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return {'available_slots': [], 'error': str(e)}

    @sync_to_async
    def _get_or_create_customer_sync(
        self, 
        phone_number: str, 
        customer_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Get or create customer using service helper for lookup.
        
        Args:
            phone_number: Customer's phone number
            customer_name: Customer's name (used if creating new customer)
            
        Returns:
            Dictionary containing customer information
        """
        # Use service to find client
        service = AppointmentBusinessBookingService(self.business_id)
        client = service.find_client_by_phone(phone_number)
        
        # Create if not found
        if not client:
            name_parts = customer_name.strip().split(maxsplit=1)
            first_name = name_parts[0] if name_parts else "Unknown"
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            client = Client.objects.create(
                phone=phone_number,
                primary_business_id=self.business_id,
                first_name=first_name,
                last_name=last_name,
                is_active=True
            )
        
        return self._serialize_client(client)

    async def get_or_create_customer(
        self, 
        phone_number: str, 
        customer_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Get existing customer or create a new one.
        
        Args:
            phone_number: Customer's phone number
            customer_name: Customer's name (used if creating new customer)
            
        Returns:
            Dictionary containing customer information
        """
        return await self._get_or_create_customer_sync(phone_number, customer_name)

    @sync_to_async
    @transaction.atomic
    def _book_appointment_sync(
        self,
        phone_number: str,
        name: str,
        date: str,
        service_ids: List[int],
        available_time_slot: Dict[str, Any],
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Book an appointment using service helper for client lookup.
        
        Args:
            phone_number: Customer's phone number
            name: Customer's full name
            date: Appointment date
            service_ids: List of service IDs to book
            available_time_slot: Selected time slot with employee_id, start_at, end_at
            notes: Additional notes for the appointment
            
        Returns:
            Dictionary containing booking confirmation details
        """
        logger.info(f"====================Booking appointment: phone={phone_number}, name={name}, date={date}, services={service_ids}====================")

        # Use service to find client
        service = AppointmentBusinessBookingService(self.business_id)
        client = service.find_client_by_phone(phone_number)
        
        # Create if not found
        if not client:
            name_parts = name.strip().split(maxsplit=1)
            first_name = name_parts[0] if name_parts else "Unknown"
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            client = Client.objects.create(
                phone=phone_number,
                primary_business_id=self.business_id,
                first_name=first_name,
                last_name=last_name,
                is_active=True
            )

        # Parse date
        appointment_date = datetime.strptime(date, '%Y-%m-%d').date()
        
        # Create appointment
        appointment = Appointment.objects.create(
            business_id=self.business_id,
            client=client,
            appointment_date=appointment_date,
            status=AppointmentStatusType.SCHEDULED,
            booking_source=BookingSourceType.AI_RECEPTIONIST,
            notes=notes or "Booked via AI receptionist",
            start_at=available_time_slot.get('start_at'),
            end_at=available_time_slot.get('end_at')
        )

        # Create appointment services
        for service_id in service_ids:
            AppointmentService.objects.create(
                appointment=appointment,
                service_id=service_id,
                staff_id=available_time_slot.get('employee_id'),
                start_at=available_time_slot.get('start_at'),
                end_at=available_time_slot.get('end_at')
            )

        logger.info(f"Booking created: Appointment ID {appointment.id}")
    
        return self._serialize_appointment(appointment)

    async def book_appointment(
        self,
        phone_number: str,
        name: str,
        date: str,
        service_ids: List[int],
        available_time_slot: Dict[str, Any],
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Book an appointment for a customer.
        
        Args:
            phone_number: Customer's phone number
            name: Customer's full name
            date: Appointment date
            service_ids: List of service IDs to book
            available_time_slot: Selected time slot with employee_id, start_at, end_at
            notes: Additional notes for the appointment
            
        Returns:
            Dictionary containing booking confirmation details
        """
        return await self._book_appointment_sync(
            phone_number, name, date, service_ids, available_time_slot, notes
        )

    @sync_to_async
    def _lookup_appointments_sync(
        self, 
        phone_number: str, 
        date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Look up customer appointments synchronously.
        
        Args:
            phone_number: Customer's phone number
            date: Optional date to filter appointments
            
        Returns:
            List of appointments
        """
        phone_number_digits = re.sub(r'[^0-9]', '', phone_number)
        logger.info(f"Looking up appointments: phone={phone_number_digits}, date={date}")

        # Build query
        query = Appointment.objects.filter(
            client__phone=phone_number_digits,
            business_id=self.business_id,
            is_active=True,
            is_deleted=False,
        ).select_related('client').prefetch_related(
            'appointment_services__service',
            'appointment_services__staff'
        ).order_by('-appointment_date')
        
        # Filter by date if provided
        if date:
            try:
                filter_date = datetime.strptime(date, '%Y-%m-%d').date()
                query = query.filter(appointment_date=filter_date)
            except ValueError:
                pass  # Invalid date format, ignore filter
        
        appointments = list(query)
        logger.info(f"Found {len(appointments)} appointments")
        
        return [self._serialize_appointment(apt) for apt in appointments]

    async def lookup_appointments(
        self, 
        phone_number: str, 
        date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Look up customer appointments.
        
        Args:
            phone_number: Customer's phone number
            date: Optional date to filter appointments
            
        Returns:
            List of appointments
        """
        return await self._lookup_appointments_sync(phone_number, date)

    @sync_to_async
    def _cancel_appointment_sync(
        self,
        appointment_id: int,
        phone_number: str,
        name: Optional[str] = None,
        service_name: Optional[str] = None,
        date: Optional[str] = None,
        time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel appointment using service helper.
        
        Args:
            appointment_id: ID of the appointment to cancel
            phone_number: Customer's phone number
            name: Customer's name (optional, for logging)
            service_name: Service name (optional, for logging)
            date: Appointment date (optional, for logging)
            time: Appointment time (optional, for logging)
            
        Returns:
            Dictionary containing cancellation confirmation
        """
        logger.info(f"Cancelling appointment: id={appointment_id}, phone={phone_number}")
        
        try:
            # Find client using service
            service = AppointmentBusinessBookingService(self.business_id)
            client = service.find_client_by_phone(phone_number)
            
            if not client:
                return {
                    'success': False,
                    'message': 'Customer not found'
                }
            
            # Use service to cancel appointment
            appointment = service.cancel_appointment(appointment_id, client.id)
            
            if not appointment:
                return {
                    'success': False,
                    'message': 'Appointment not found or already cancelled'
                }
            
            logger.info(f"Appointment {appointment_id} cancelled successfully")
            
            return {
                'success': True,
                'message': 'Appointment cancelled successfully',
                'appointment': self._serialize_appointment(appointment)
            }
            
        except Exception as e:
            logger.error(f"Error canceling appointment: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    async def cancel_appointment(
        self,
        appointment_id: int,
        phone_number: str,
        name: Optional[str] = None,
        service_name: Optional[str] = None,
        date: Optional[str] = None,
        time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel a customer appointment.
        
        Args:
            appointment_id: ID of the appointment to cancel
            phone_number: Customer's phone number
            name: Customer's name (optional, for logging)
            service_name: Service name (optional, for logging)
            date: Appointment date (optional, for logging)
            time: Appointment time (optional, for logging)
            
        Returns:
            Dictionary containing cancellation confirmation
        """
        return await self._cancel_appointment_sync(
            appointment_id, phone_number, name, service_name, date, time
        )

    @sync_to_async
    def _search_services_sync(self, service_name: str) -> List[Dict[str, Any]]:
        """
        Search for services by name synchronously.
        
        Args:
            service_name: Name of the service to search for
            
        Returns:
            List of matching services
        """
        services = Service.objects.filter(
            business_id=self.business_id,
            name__icontains=service_name,
            is_active=True
        ).select_related('category')
        
        return [self._serialize_service(service) for service in services]

    async def search_services(self, service_name: str) -> List[Dict[str, Any]]:
        """
        Search for services by name.
        
        Args:
            service_name: Name of the service to search for
            
        Returns:
            List of matching services
        """
        return await self._search_services_sync(service_name)
    
    @sync_to_async
    def get_staff_information(self, staff_name: str) -> Dict[str, Any]:
        """Get information for a specific staff by staff name."""
        staff = Staff.objects.filter(first_name__icontains=staff_name).first()
        return self._serialize_staff(staff)

    # Serialization helper methods
    
    def _serialize_business(self, business: Business) -> Dict[str, Any]:
        """Serialize Business model to dictionary."""
        serialize = BusinessSerializer(business).data if business else None
        return serialize

    def _serialize_service(self, service: Service) -> Dict[str, Any]:
        """Serialize Service model to dictionary."""
        
        serialize = ServiceSerializer(service).data if service else None
        return serialize

    def _serialize_client(self, client: Client) -> Dict[str, Any]:
        """Serialize Client model to dictionary."""
        serialize = ClientSerializer(client).data if client else None
        return serialize

    def _serialize_appointment(self, appointment: Appointment) -> Dict[str, Any]:
        """Serialize Appointment model to dictionary."""
        # Get appointment services
        serialize = AppointmentDetailSerializer(appointment).data
        serialize['business'] = str(serialize['business'])
        
        logger.info(f"Serialized appointment: {serialize}")
        return serialize

    def _serialize_staff(self, staff: Staff) -> Dict[str, Any]:
        """Serialize Staff model to dictionary."""
        serialize = StaffSerializer(staff).data if staff else None
        return serialize
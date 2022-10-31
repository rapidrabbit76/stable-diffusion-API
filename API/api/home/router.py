from datetime import datetime
from fastapi import Response
from fastapi_restful.cbv import cbv
from fastapi_restful.inferring_router import InferringRouter

from core.settings import get_settings


router = InferringRouter()
env = get_settings()


@cbv(router)
class Home:
    @router.get("/")
    async def index(self):
        """ELB check"""
        current_time = datetime.utcnow()
        msg = f"Notification API (UTC: {current_time.strftime('%Y.%m.%d %H:%M:%S')})"
        return Response(msg)

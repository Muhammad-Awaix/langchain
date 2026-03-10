from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class User(BaseModel):
    name: str ="Awais"
    age: Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0, lt=4.0, description="CGPA must be between 0 and 4.0")

new_user = User(age=25, email='mawais.ai021@gmail.com', cgpa=3.5)
print(new_user)
# ------------------------------------------------------------------------
# Class CoursesUpdated
#
# Copyright   2024 Pimenko <support@pimenko.com><pimenko.com>
# Author      Jordan Kesraoui
# License     https://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
# ------------------------------------------------------------------------
from pydantic import BaseModel

class CoursesUpdated(BaseModel):
    token: str
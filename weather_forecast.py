"""
Krishimitra - offline (non-API) weather forecasting engine.

How it works (no internet / no weather API is used anywhere):
  1. Every supported city is mapped to a climate zone that stores 12-month
     climatological normals (max/min temperature, rainfall, rain-days,
     humidity, wind) for that part of India.
  2. Normals are smoothly interpolated to the exact date and corrected for the
     city's elevation (lapse rate).
  3. Day-to-day "weather" (hot/cool spells, wet/dry spells) is produced by a
     deterministic, auto-correlated stochastic weather generator, so the same
     city + date always gives the same forecast and consecutive days look
     like real weather instead of random jumps.
  4. Sunrise, sunset, day length and UV index are calculated with astronomy
     formulas from latitude / longitude / date.

NOTE: this is a climatology-based outlook, not a satellite/NWP forecast.
"""
import hashlib
import math
import difflib
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

IST = timezone(timedelta(hours=5, minutes=30))
_ND = NormalDist()

# --------------------------------------------------------------------------
# Climate zones: 12 monthly normals (Jan..Dec) at the zone reference elevation
# --------------------------------------------------------------------------
ZONES = {
    "north_plains": dict(
        ref_elev=216, sigma=(3.0, 2.3),
        tmax=[21, 24, 30, 37, 40, 39, 35, 34, 34, 33, 28, 23],
        tmin=[8, 11, 16, 22, 26, 28, 27, 26, 24, 19, 13, 9],
        rain=[19, 18, 13, 8, 14, 65, 210, 247, 123, 15, 4, 10],
        rdays=[2, 2, 2, 1, 2, 4, 10, 11, 6, 1, 0.5, 1],
        hum=[70, 60, 50, 38, 38, 50, 72, 78, 68, 55, 58, 68],
        wind=[6, 7, 8, 9, 10, 10, 8, 6, 6, 5, 4, 5], fog=True),
    "gangetic": dict(
        ref_elev=111, sigma=(3.0, 2.2),
        tmax=[23, 27, 33, 39, 41, 39, 34, 33, 33, 33, 29, 25],
        tmin=[8, 11, 16, 22, 26, 28, 27, 27, 25, 20, 13, 9],
        rain=[20, 20, 10, 6, 12, 90, 300, 330, 190, 40, 5, 5],
        rdays=[2, 2, 1, 1, 2, 5, 13, 14, 8, 2, 0.4, 0.5],
        hum=[75, 65, 50, 40, 40, 55, 78, 82, 76, 66, 66, 74],
        wind=[5, 6, 7, 8, 8, 8, 6, 5, 5, 4, 3, 4], fog=True),
    "arid": dict(
        ref_elev=231, sigma=(2.5, 2.0),
        tmax=[25, 28, 34, 40, 43, 41, 36, 34, 36, 37, 31, 27],
        tmin=[9, 12, 18, 24, 28, 29, 27, 26, 24, 19, 13, 9],
        rain=[3, 4, 4, 3, 9, 31, 114, 141, 48, 5, 2, 2],
        rdays=[0.2, 0.3, 0.3, 0.2, 0.8, 2, 6, 7, 3, 0.3, 0.1, 0.1],
        hum=[55, 45, 35, 27, 28, 40, 62, 68, 55, 35, 40, 52],
        wind=[6, 7, 9, 11, 13, 15, 13, 10, 8, 5, 5, 5], fog=False),
    "gujarat": dict(
        ref_elev=53, sigma=(2.2, 1.9),
        tmax=[29, 32, 37, 41, 42, 38, 33, 32, 34, 36, 33, 30],
        tmin=[13, 16, 21, 25, 28, 28, 26, 25, 25, 22, 17, 13],
        rain=[3, 1, 1, 2, 13, 98, 310, 226, 142, 16, 5, 1],
        rdays=[0.2, 0.1, 0.1, 0.2, 0.9, 5, 13, 11, 7, 1, 0.3, 0.1],
        hum=[50, 42, 36, 35, 45, 62, 78, 80, 70, 52, 48, 50],
        wind=[7, 8, 9, 11, 12, 13, 12, 10, 8, 6, 6, 6], fog=False),
    "east_plains": dict(
        ref_elev=6, sigma=(1.8, 1.5),
        tmax=[27, 30, 34, 36, 36, 34, 32, 32, 32, 32, 30, 27],
        tmin=[13, 16, 21, 25, 26, 27, 26, 26, 26, 24, 18, 13],
        rain=[10, 31, 36, 43, 140, 297, 325, 328, 252, 114, 20, 5],
        rdays=[1, 2, 3, 4, 8, 14, 19, 19, 14, 6, 1, 0.4],
        hum=[73, 68, 66, 70, 74, 80, 85, 86, 85, 80, 73, 72],
        wind=[5, 6, 8, 9, 9, 9, 8, 7, 6, 5, 4, 4], fog=True),
    "northeast": dict(
        ref_elev=55, sigma=(1.8, 1.5),
        tmax=[24, 26, 30, 31, 31, 32, 32, 32, 32, 31, 28, 25],
        tmin=[11, 13, 17, 20, 22, 24, 25, 25, 24, 21, 16, 12],
        rain=[12, 18, 52, 125, 215, 310, 355, 270, 215, 100, 20, 5],
        rdays=[2, 3, 6, 11, 15, 18, 19, 16, 14, 7, 2, 1],
        hum=[80, 74, 68, 74, 80, 85, 86, 86, 85, 82, 80, 80],
        wind=[3, 4, 5, 6, 5, 5, 4, 4, 4, 3, 3, 3], fog=True),
    "central": dict(
        ref_elev=310, sigma=(2.6, 2.2),
        tmax=[29, 32, 37, 41, 43, 38, 31, 30, 32, 32, 30, 28],
        tmin=[13, 15, 19, 24, 28, 27, 24, 24, 23, 20, 15, 12],
        rain=[14, 21, 22, 12, 19, 185, 330, 240, 160, 60, 17, 9],
        rdays=[1, 2, 2, 1, 2, 9, 16, 15, 9, 3, 1, 0.6],
        hum=[55, 45, 35, 30, 30, 55, 78, 80, 75, 60, 52, 55],
        wind=[5, 6, 7, 8, 9, 10, 8, 7, 6, 4, 4, 4], fog=False),
    "deccan": dict(
        ref_elev=540, sigma=(2.0, 2.0),
        tmax=[29, 32, 35, 38, 39, 34, 31, 30, 31, 31, 29, 28],
        tmin=[15, 17, 21, 24, 26, 23, 22, 21, 21, 20, 17, 14],
        rain=[4, 9, 12, 25, 32, 100, 160, 160, 160, 100, 32, 6],
        rdays=[0.3, 0.8, 1, 2, 3, 7, 11, 11, 9, 5, 2, 0.4],
        hum=[55, 45, 40, 40, 42, 62, 72, 75, 72, 65, 58, 55],
        wind=[8, 8, 8, 9, 10, 12, 12, 11, 9, 7, 7, 7], fog=False),
    "south_interior": dict(
        ref_elev=920, sigma=(1.5, 1.5),
        tmax=[27, 30, 33, 34, 33, 29, 27, 27, 28, 28, 27, 26],
        tmin=[15, 17, 19, 21, 21, 20, 19, 19, 19, 19, 17, 15],
        rain=[2, 5, 6, 45, 105, 80, 110, 140, 180, 150, 60, 15],
        rdays=[0.2, 0.4, 0.7, 4, 8, 8, 10, 12, 12, 10, 5, 1],
        hum=[62, 55, 48, 50, 60, 72, 76, 77, 76, 75, 70, 65],
        wind=[8, 8, 8, 8, 9, 10, 11, 10, 8, 7, 7, 7], fog=False),
    "west_coast": dict(
        ref_elev=14, sigma=(1.2, 1.1),
        tmax=[31, 32, 33, 33, 34, 32, 30, 29, 30, 33, 34, 33],
        tmin=[19, 20, 23, 26, 28, 27, 26, 25, 25, 25, 23, 21],
        rain=[3, 1, 1, 1, 12, 520, 709, 440, 297, 86, 14, 3],
        rdays=[0.1, 0.1, 0.1, 0.2, 1, 16, 24, 23, 17, 5, 1, 0.2],
        hum=[65, 65, 68, 70, 72, 82, 88, 88, 84, 75, 66, 64],
        wind=[9, 9, 10, 11, 12, 16, 17, 15, 11, 8, 8, 8], fog=False),
    "sw_coast": dict(
        ref_elev=3, sigma=(1.0, 1.0),
        tmax=[32, 33, 33, 33, 32, 29, 28, 29, 30, 31, 31, 32],
        tmin=[22, 23, 25, 26, 26, 24, 23, 23, 23, 23, 23, 22],
        rain=[12, 18, 38, 120, 330, 740, 690, 420, 260, 290, 180, 50],
        rdays=[1, 1, 3, 7, 14, 23, 25, 22, 15, 13, 9, 3],
        hum=[70, 70, 73, 75, 80, 88, 90, 88, 84, 82, 78, 72],
        wind=[8, 8, 8, 8, 8, 10, 10, 9, 7, 6, 6, 7], fog=False),
    "east_coast_south": dict(
        ref_elev=7, sigma=(1.3, 1.3),
        tmax=[29, 31, 33, 35, 38, 37, 35, 34, 34, 32, 29, 28],
        tmin=[20, 21, 23, 26, 27, 27, 26, 25, 25, 24, 22, 21],
        rain=[24, 7, 15, 25, 52, 53, 83, 124, 118, 267, 309, 139],
        rdays=[2, 1, 1, 2, 3, 6, 9, 11, 9, 13, 12, 7],
        hum=[68, 68, 68, 70, 65, 60, 63, 67, 70, 76, 78, 74],
        wind=[10, 11, 12, 13, 14, 14, 13, 12, 11, 9, 10, 10], fog=False),
    "east_coast_north": dict(
        ref_elev=10, sigma=(1.4, 1.4),
        tmax=[29, 31, 33, 34, 35, 33, 31, 31, 31, 31, 29, 28],
        tmin=[18, 20, 23, 26, 27, 26, 25, 25, 25, 24, 20, 17],
        rain=[10, 15, 15, 25, 55, 130, 200, 215, 200, 260, 70, 10],
        rdays=[1, 1, 1, 2, 3, 8, 12, 12, 11, 10, 3, 1],
        hum=[70, 70, 70, 72, 72, 75, 78, 78, 78, 76, 70, 68],
        wind=[8, 8, 9, 10, 11, 12, 11, 10, 9, 8, 8, 8], fog=False),
    "himalaya": dict(
        ref_elev=2200, sigma=(3.0, 2.8),
        tmax=[10, 11, 16, 21, 24, 24, 21, 20, 20, 19, 15, 12],
        tmin=[2, 3, 7, 11, 14, 16, 16, 16, 14, 10, 6, 3],
        rain=[60, 60, 55, 50, 75, 190, 430, 420, 190, 40, 15, 25],
        rdays=[4, 5, 5, 5, 7, 12, 20, 20, 11, 3, 2, 3],
        hum=[60, 60, 55, 50, 50, 65, 85, 88, 78, 60, 55, 58],
        wind=[8, 9, 10, 11, 11, 10, 8, 7, 7, 7, 7, 7], fog=True),
    "kashmir": dict(
        ref_elev=1585, sigma=(3.2, 2.8),
        tmax=[5, 8, 14, 20, 25, 30, 31, 30, 28, 22, 14, 8],
        tmin=[-2, 0, 5, 9, 12, 16, 19, 18, 13, 6, 1, -1],
        rain=[70, 85, 115, 105, 70, 35, 60, 65, 35, 35, 20, 40],
        rdays=[6, 7, 10, 10, 8, 5, 8, 8, 4, 3, 2, 4],
        hum=[78, 72, 65, 58, 55, 50, 65, 68, 62, 60, 68, 78],
        wind=[5, 6, 7, 7, 7, 6, 6, 5, 5, 4, 4, 4], fog=True),
    "ladakh": dict(
        ref_elev=3500, sigma=(3.5, 3.0),
        tmax=[-2, 2, 8, 14, 19, 24, 27, 26, 22, 15, 8, 1],
        tmin=[-14, -11, -5, 0, 4, 8, 12, 11, 6, -1, -7, -12],
        rain=[10, 8, 10, 8, 8, 5, 15, 15, 10, 4, 2, 5],
        rdays=[1, 1, 1, 1, 1, 1, 2, 2, 1, 0.3, 0.2, 1],
        hum=[40, 38, 32, 28, 28, 30, 40, 42, 35, 30, 35, 42],
        wind=[10, 11, 13, 14, 14, 14, 12, 11, 10, 9, 8, 9], fog=False),
}

# --------------------------------------------------------------------------
# City database: name -> (state, lat, lon, elevation_m, zone, temp_offset, rain_multiplier)
# --------------------------------------------------------------------------
_RAW = [
    # Andhra Pradesh
    ("Visakhapatnam", "Andhra Pradesh", 17.69, 83.22, 3, "east_coast_north", 0, 1.0),
    ("Vijayawada", "Andhra Pradesh", 16.51, 80.65, 11, "east_coast_north", 1.5, 0.8),
    ("Guntur", "Andhra Pradesh", 16.31, 80.44, 33, "east_coast_north", 1.5, 0.7),
    ("Tirupati", "Andhra Pradesh", 13.63, 79.42, 161, "east_coast_south", 0.5, 1.0),
    ("Kurnool", "Andhra Pradesh", 15.83, 78.04, 280, "deccan", 1.0, 0.8),
    ("Nellore", "Andhra Pradesh", 14.44, 79.99, 18, "east_coast_south", 0, 1.0),
    ("Anantapur", "Andhra Pradesh", 14.68, 77.60, 350, "deccan", 0.5, 0.55),
    ("Rajahmundry", "Andhra Pradesh", 17.00, 81.80, 14, "east_coast_north", 0, 1.1),
    ("Kakinada", "Andhra Pradesh", 16.99, 82.25, 2, "east_coast_north", 0, 1.1),
    ("Kadapa", "Andhra Pradesh", 14.47, 78.82, 138, "deccan", 1.5, 0.8),
    # Arunachal / Assam / NE
    ("Itanagar", "Arunachal Pradesh", 27.08, 93.61, 320, "northeast", 0, 1.0),
    ("Guwahati", "Assam", 26.14, 91.74, 55, "northeast", 0, 1.0),
    ("Dibrugarh", "Assam", 27.48, 94.91, 105, "northeast", -1, 1.1),
    ("Silchar", "Assam", 24.83, 92.78, 28, "northeast", 0, 1.2),
    ("Jorhat", "Assam", 26.75, 94.20, 86, "northeast", 0, 1.0),
    ("Imphal", "Manipur", 24.82, 93.94, 786, "northeast", 3, 0.75),
    ("Shillong", "Meghalaya", 25.57, 91.88, 1496, "northeast", 0, 1.0),
    ("Aizawl", "Mizoram", 23.73, 92.72, 1132, "northeast", 1.5, 1.0),
    ("Kohima", "Nagaland", 25.67, 94.11, 1444, "northeast", 1.5, 1.0),
    ("Agartala", "Tripura", 23.83, 91.28, 13, "northeast", 1.5, 0.8),
    ("Gangtok", "Sikkim", 27.33, 88.61, 1650, "northeast", 0, 1.0),
    # Bihar
    ("Patna", "Bihar", 25.59, 85.14, 53, "gangetic", 0, 1.0),
    ("Gaya", "Bihar", 24.80, 85.00, 111, "gangetic", 0.5, 1.0),
    ("Bhagalpur", "Bihar", 25.24, 86.97, 52, "gangetic", 0, 1.1),
    ("Muzaffarpur", "Bihar", 26.12, 85.39, 60, "gangetic", 0, 1.2),
    ("Darbhanga", "Bihar", 26.15, 85.90, 52, "gangetic", 0, 1.2),
    # Chandigarh / Delhi / Haryana / Punjab
    ("Chandigarh", "Chandigarh", 30.73, 76.78, 321, "north_plains", -0.5, 1.3),
    ("Delhi", "Delhi", 28.61, 77.21, 216, "north_plains", 0, 1.0),
    ("Gurugram", "Haryana", 28.46, 77.03, 217, "north_plains", 0, 1.0),
    ("Faridabad", "Haryana", 28.41, 77.31, 198, "north_plains", 0, 1.0),
    ("Ambala", "Haryana", 30.38, 76.78, 264, "north_plains", 0, 1.3),
    ("Hisar", "Haryana", 29.15, 75.72, 215, "north_plains", 0, 0.8),
    ("Rohtak", "Haryana", 28.89, 76.61, 220, "north_plains", 0, 0.9),
    ("Karnal", "Haryana", 29.69, 76.99, 232, "north_plains", 0, 1.1),
    ("Ludhiana", "Punjab", 30.90, 75.85, 244, "north_plains", -0.5, 1.2),
    ("Amritsar", "Punjab", 31.63, 74.87, 234, "north_plains", -1, 1.2),
    ("Patiala", "Punjab", 30.34, 76.39, 250, "north_plains", 0, 1.3),
    ("Jalandhar", "Punjab", 31.33, 75.58, 228, "north_plains", 0, 1.2),
    ("Bathinda", "Punjab", 30.21, 74.95, 211, "north_plains", 0.5, 0.8),
    # Chhattisgarh / Jharkhand / MP
    ("Raipur", "Chhattisgarh", 21.25, 81.63, 298, "central", 0, 1.0),
    ("Bilaspur", "Chhattisgarh", 22.08, 82.15, 264, "central", 0, 1.0),
    ("Jagdalpur", "Chhattisgarh", 19.08, 82.02, 553, "central", -1.5, 1.2),
    ("Ranchi", "Jharkhand", 23.34, 85.31, 651, "central", -1, 1.3),
    ("Jamshedpur", "Jharkhand", 22.80, 86.20, 135, "central", -2, 1.3),
    ("Dhanbad", "Jharkhand", 23.80, 86.43, 222, "central", -1.5, 1.3),
    ("Bhopal", "Madhya Pradesh", 23.26, 77.41, 527, "central", -0.5, 1.0),
    ("Indore", "Madhya Pradesh", 22.72, 75.86, 553, "central", -0.5, 0.95),
    ("Gwalior", "Madhya Pradesh", 26.22, 78.18, 197, "north_plains", 1, 0.95),
    ("Jabalpur", "Madhya Pradesh", 23.18, 79.99, 412, "central", 0, 1.3),
    ("Ujjain", "Madhya Pradesh", 23.18, 75.78, 491, "central", 0, 0.95),
    ("Sagar", "Madhya Pradesh", 23.84, 78.74, 523, "central", 0, 1.2),
    ("Rewa", "Madhya Pradesh", 24.53, 81.30, 327, "central", 1, 1.1),
    # Goa / Gujarat
    ("Panaji", "Goa", 15.50, 73.83, 7, "west_coast", 0, 1.3),
    ("Ahmedabad", "Gujarat", 23.02, 72.57, 53, "gujarat", 0, 1.0),
    ("Surat", "Gujarat", 21.17, 72.83, 13, "gujarat", -1.5, 1.5),
    ("Vadodara", "Gujarat", 22.31, 73.18, 35, "gujarat", 0, 1.1),
    ("Rajkot", "Gujarat", 22.30, 70.80, 138, "gujarat", 0, 0.8),
    ("Bhavnagar", "Gujarat", 21.77, 72.15, 5, "gujarat", -1.5, 0.8),
    ("Jamnagar", "Gujarat", 22.47, 70.07, 21, "gujarat", -1.5, 0.8),
    ("Bhuj", "Gujarat", 23.25, 69.67, 110, "arid", -1, 1.1),
    ("Junagadh", "Gujarat", 21.52, 70.46, 107, "gujarat", -1, 1.4),
    # Himachal / J&K / Ladakh / Uttarakhand
    ("Shimla", "Himachal Pradesh", 31.10, 77.17, 2206, "himalaya", 0, 1.0),
    ("Manali", "Himachal Pradesh", 32.24, 77.19, 2050, "himalaya", 0, 0.9),
    ("Dharamshala", "Himachal Pradesh", 32.22, 76.32, 1457, "himalaya", 0, 2.5),
    ("Solan", "Himachal Pradesh", 30.91, 77.10, 1550, "himalaya", 0, 1.0),
    ("Mandi", "Himachal Pradesh", 31.70, 76.93, 760, "himalaya", 0, 0.9),
    ("Srinagar", "Jammu & Kashmir", 34.08, 74.80, 1585, "kashmir", 0, 1.0),
    ("Jammu", "Jammu & Kashmir", 32.73, 74.87, 327, "north_plains", -1, 1.7),
    ("Leh", "Ladakh", 34.16, 77.58, 3500, "ladakh", 0, 1.0),
    ("Dehradun", "Uttarakhand", 30.32, 78.03, 640, "north_plains", 1, 2.0),
    ("Haridwar", "Uttarakhand", 29.95, 78.16, 249, "north_plains", 0, 1.5),
    ("Nainital", "Uttarakhand", 29.38, 79.46, 2084, "himalaya", 0, 1.6),
    ("Rishikesh", "Uttarakhand", 30.10, 78.29, 372, "north_plains", 0, 1.7),
    ("Haldwani", "Uttarakhand", 29.22, 79.51, 424, "north_plains", 0, 1.6),
    ("Mussoorie", "Uttarakhand", 30.46, 78.07, 2005, "himalaya", 0, 1.7),
    # Karnataka
    ("Bengaluru", "Karnataka", 12.97, 77.59, 920, "south_interior", 0, 1.0),
    ("Mysuru", "Karnataka", 12.30, 76.64, 770, "south_interior", 0, 0.8),
    ("Hubballi", "Karnataka", 15.36, 75.12, 670, "deccan", 0, 0.8),
    ("Mangaluru", "Karnataka", 12.91, 74.86, 22, "sw_coast", 0, 1.0),
    ("Belagavi", "Karnataka", 15.85, 74.50, 751, "deccan", 0, 1.2),
    ("Kalaburagi", "Karnataka", 17.33, 76.83, 454, "deccan", 2, 0.8),
    ("Davanagere", "Karnataka", 14.46, 75.92, 575, "deccan", 0, 0.7),
    ("Shivamogga", "Karnataka", 13.93, 75.57, 570, "south_interior", 1, 1.3),
    ("Tumakuru", "Karnataka", 13.34, 77.10, 822, "south_interior", 0.5, 0.8),
    ("Ballari", "Karnataka", 15.14, 76.92, 449, "deccan", 2.5, 0.7),
    ("Hassan", "Karnataka", 13.00, 76.10, 950, "south_interior", -0.5, 1.0),
    ("Udupi", "Karnataka", 13.34, 74.75, 6, "sw_coast", 0, 1.0),
    ("Mandya", "Karnataka", 12.52, 76.90, 678, "south_interior", 1.5, 0.8),
    ("Bidar", "Karnataka", 17.91, 77.52, 664, "deccan", -0.5, 1.0),
    ("Vijayapura", "Karnataka", 16.83, 75.71, 593, "deccan", 1.5, 0.6),
    ("Raichur", "Karnataka", 16.20, 77.35, 389, "deccan", 2, 0.6),
    # Kerala
    ("Thiruvananthapuram", "Kerala", 8.52, 76.94, 10, "sw_coast", 0, 0.8),
    ("Kochi", "Kerala", 9.93, 76.27, 3, "sw_coast", 0, 1.0),
    ("Kozhikode", "Kerala", 11.25, 75.78, 1, "sw_coast", 0, 1.0),
    ("Thrissur", "Kerala", 10.52, 76.21, 2, "sw_coast", 0, 1.0),
    ("Kollam", "Kerala", 8.89, 76.61, 8, "sw_coast", 0, 0.8),
    ("Kannur", "Kerala", 11.87, 75.37, 5, "sw_coast", 0, 1.05),
    ("Palakkad", "Kerala", 10.78, 76.65, 84, "sw_coast", 2, 0.85),
    ("Alappuzha", "Kerala", 9.49, 76.33, 1, "sw_coast", 0, 1.0),
    ("Kottayam", "Kerala", 9.59, 76.52, 3, "sw_coast", 0, 1.0),
    # Maharashtra
    ("Mumbai", "Maharashtra", 19.08, 72.88, 14, "west_coast", 0, 1.0),
    ("Pune", "Maharashtra", 18.52, 73.86, 560, "deccan", 0, 0.9),
    ("Nagpur", "Maharashtra", 21.15, 79.09, 310, "central", 0, 1.0),
    ("Nashik", "Maharashtra", 19.99, 73.79, 565, "deccan", -0.5, 1.0),
    ("Aurangabad", "Maharashtra", 19.88, 75.34, 568, "deccan", 0, 0.85),
    ("Solapur", "Maharashtra", 17.66, 75.91, 458, "deccan", 2, 0.8),
    ("Kolhapur", "Maharashtra", 16.70, 74.24, 550, "deccan", -0.5, 1.8),
    ("Amravati", "Maharashtra", 20.93, 77.75, 343, "central", 0.5, 1.0),
    ("Nanded", "Maharashtra", 19.15, 77.32, 358, "deccan", 0.5, 1.0),
    ("Akola", "Maharashtra", 20.71, 77.00, 282, "central", 0.5, 0.9),
    ("Ratnagiri", "Maharashtra", 16.99, 73.30, 11, "west_coast", -0.5, 1.05),
    ("Thane", "Maharashtra", 19.20, 72.97, 10, "west_coast", 0, 1.1),
    ("Sangli", "Maharashtra", 16.85, 74.57, 549, "deccan", 0.5, 1.0),
    ("Jalgaon", "Maharashtra", 21.00, 75.57, 209, "central", 1.5, 0.9),
    ("Latur", "Maharashtra", 18.40, 76.58, 636, "deccan", 0, 1.0),
    # Odisha / Puducherry / Andaman
    ("Bhubaneswar", "Odisha", 20.30, 85.82, 45, "east_coast_north", 0.5, 1.1),
    ("Cuttack", "Odisha", 20.46, 85.88, 36, "east_coast_north", 0.5, 1.1),
    ("Puri", "Odisha", 19.81, 85.83, 3, "east_coast_north", -0.5, 1.1),
    ("Sambalpur", "Odisha", 21.47, 83.97, 138, "central", 1, 1.1),
    ("Rourkela", "Odisha", 22.26, 84.85, 219, "central", 0, 1.2),
    ("Berhampur", "Odisha", 19.31, 84.79, 25, "east_coast_north", 0.5, 1.0),
    ("Puducherry", "Puducherry", 11.94, 79.81, 3, "east_coast_south", 0, 1.1),
    ("Port Blair", "Andaman & Nicobar", 11.62, 92.73, 16, "sw_coast", 0, 0.85),
    # Rajasthan
    ("Jaipur", "Rajasthan", 26.91, 75.79, 431, "arid", -1, 1.6),
    ("Jodhpur", "Rajasthan", 26.24, 73.02, 231, "arid", 0, 1.0),
    ("Udaipur", "Rajasthan", 24.58, 73.71, 598, "arid", -1.5, 2.1),
    ("Kota", "Rajasthan", 25.21, 75.86, 271, "arid", 0, 2.0),
    ("Ajmer", "Rajasthan", 26.45, 74.64, 486, "arid", -1.5, 1.8),
    ("Bikaner", "Rajasthan", 28.02, 73.31, 224, "arid", 0.5, 0.65),
    ("Jaisalmer", "Rajasthan", 26.92, 70.90, 225, "arid", 0.5, 0.5),
    ("Alwar", "Rajasthan", 27.55, 76.63, 271, "north_plains", 0, 1.0),
    ("Sikar", "Rajasthan", 27.61, 75.14, 431, "arid", 0, 1.5),
    ("Sri Ganganagar", "Rajasthan", 29.92, 73.88, 178, "arid", -0.5, 1.1),
    ("Bharatpur", "Rajasthan", 27.22, 77.49, 174, "north_plains", 0, 1.0),
    # Tamil Nadu
    ("Chennai", "Tamil Nadu", 13.08, 80.27, 7, "east_coast_south", 0, 1.0),
    ("Coimbatore", "Tamil Nadu", 11.00, 76.96, 411, "south_interior", 0, 0.9),
    ("Madurai", "Tamil Nadu", 9.93, 78.12, 101, "east_coast_south", 2, 1.1),
    ("Tiruchirappalli", "Tamil Nadu", 10.79, 78.69, 88, "east_coast_south", 2, 0.9),
    ("Salem", "Tamil Nadu", 11.66, 78.15, 278, "south_interior", 0, 0.9),
    ("Tirunelveli", "Tamil Nadu", 8.73, 77.70, 47, "east_coast_south", 1.5, 0.8),
    ("Thanjavur", "Tamil Nadu", 10.79, 79.14, 60, "east_coast_south", 1, 1.1),
    ("Vellore", "Tamil Nadu", 12.92, 79.13, 216, "east_coast_south", 3, 1.2),
    ("Erode", "Tamil Nadu", 11.34, 77.72, 183, "south_interior", 0, 0.7),
    ("Thoothukudi", "Tamil Nadu", 8.76, 78.13, 3, "east_coast_south", 0.5, 0.8),
    ("Kanyakumari", "Tamil Nadu", 8.08, 77.54, 5, "sw_coast", -1, 0.5),
    ("Ooty", "Tamil Nadu", 11.41, 76.69, 2240, "south_interior", -1.5, 1.5),
    # Telangana
    ("Hyderabad", "Telangana", 17.39, 78.49, 542, "deccan", 0, 1.0),
    ("Warangal", "Telangana", 17.97, 79.59, 266, "deccan", 2.5, 1.1),
    ("Nizamabad", "Telangana", 18.67, 78.10, 395, "deccan", 2, 1.0),
    ("Karimnagar", "Telangana", 18.44, 79.13, 266, "deccan", 2.5, 1.0),
    ("Khammam", "Telangana", 17.25, 80.15, 121, "deccan", 3, 1.0),
    # Uttar Pradesh
    ("Lucknow", "Uttar Pradesh", 26.85, 80.95, 123, "gangetic", 0, 1.0),
    ("Kanpur", "Uttar Pradesh", 26.45, 80.35, 126, "gangetic", 0.5, 0.95),
    ("Varanasi", "Uttar Pradesh", 25.32, 83.00, 81, "gangetic", 0, 1.0),
    ("Agra", "Uttar Pradesh", 27.18, 78.02, 169, "north_plains", 1, 0.9),
    ("Prayagraj", "Uttar Pradesh", 25.43, 81.85, 98, "gangetic", 0.5, 1.0),
    ("Meerut", "Uttar Pradesh", 28.98, 77.71, 222, "north_plains", 0, 1.2),
    ("Ghaziabad", "Uttar Pradesh", 28.67, 77.45, 214, "north_plains", 0, 1.0),
    ("Noida", "Uttar Pradesh", 28.54, 77.39, 200, "north_plains", 0, 1.0),
    ("Gorakhpur", "Uttar Pradesh", 26.76, 83.37, 84, "gangetic", -0.5, 1.3),
    ("Bareilly", "Uttar Pradesh", 28.37, 79.43, 172, "gangetic", -0.5, 1.1),
    ("Aligarh", "Uttar Pradesh", 27.88, 78.08, 178, "north_plains", 0, 1.0),
    ("Moradabad", "Uttar Pradesh", 28.84, 78.78, 186, "north_plains", 0, 1.2),
    ("Jhansi", "Uttar Pradesh", 25.45, 78.57, 285, "north_plains", 1.5, 1.1),
    ("Mathura", "Uttar Pradesh", 27.49, 77.67, 174, "north_plains", 0, 1.0),
    ("Saharanpur", "Uttar Pradesh", 29.96, 77.55, 269, "north_plains", 0, 1.5),
    ("Ayodhya", "Uttar Pradesh", 26.80, 82.20, 93, "gangetic", 0, 1.1),
    # West Bengal
    ("Kolkata", "West Bengal", 22.57, 88.36, 6, "east_plains", 0, 1.0),
    ("Howrah", "West Bengal", 22.59, 88.26, 12, "east_plains", 0, 1.0),
    ("Siliguri", "West Bengal", 26.72, 88.43, 122, "northeast", 1.5, 1.6),
    ("Durgapur", "West Bengal", 23.55, 87.32, 65, "east_plains", 1.5, 0.9),
    ("Asansol", "West Bengal", 23.68, 86.98, 115, "east_plains", 1.5, 0.9),
    ("Darjeeling", "West Bengal", 27.04, 88.26, 2042, "northeast", 0.5, 1.1),
    ("Kharagpur", "West Bengal", 22.35, 87.32, 30, "east_plains", 1, 0.95),
    ("Bardhaman", "West Bengal", 23.23, 87.86, 30, "east_plains", 1, 0.95),
    ("Berhampore", "West Bengal", 24.10, 88.25, 19, "east_plains", 1, 0.85),
]

CITIES = {}
for _n, _s, _la, _lo, _el, _z, _dt, _rm in _RAW:
    CITIES[_n.lower()] = dict(name=_n, state=_s, lat=_la, lon=_lo, elev=_el,
                              zone=_z, dt=_dt, rm=_rm)

ALIASES = {
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "mysore": "mysuru", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "poona": "pune", "vizag": "visakhapatnam", "gurgaon": "gurugram",
    "baroda": "vadodara", "banaras": "varanasi", "benares": "varanasi",
    "allahabad": "prayagraj", "belgaum": "belagavi", "bellary": "ballari",
    "bijapur": "vijayapura", "gulbarga": "kalaburagi", "hubli": "hubballi",
    "mangalore": "mangaluru", "shimoga": "shivamogga", "tumkur": "tumakuru",
    "calicut": "kozhikode", "trichy": "tiruchirappalli", "tuticorin": "thoothukudi",
    "pondicherry": "puducherry", "ooty (udhagamandalam)": "ooty", "udhagamandalam": "ooty",
    "bhubaneshwar": "bhubaneswar", "new delhi": "delhi", "alleppey": "alappuzha",
    "thiruvananthapuram": "thiruvananthapuram", "trichur": "thrissur", "raichur": "raichur",
    "aurangabad (maharashtra)": "aurangabad", "dharamsala": "dharamshala",
    "vijayawada": "vijayawada", "banglore": "bengaluru", "bengalore": "bengaluru",
}

STATE_CAPITALS = {
    "andhra pradesh": "vijayawada", "arunachal pradesh": "itanagar", "assam": "guwahati",
    "bihar": "patna", "chhattisgarh": "raipur", "goa": "panaji", "gujarat": "ahmedabad",
    "haryana": "chandigarh", "himachal pradesh": "shimla", "jharkhand": "ranchi",
    "karnataka": "bengaluru", "kerala": "thiruvananthapuram", "madhya pradesh": "bhopal",
    "maharashtra": "mumbai", "manipur": "imphal", "meghalaya": "shillong",
    "mizoram": "aizawl", "nagaland": "kohima", "odisha": "bhubaneswar", "orissa": "bhubaneswar",
    "punjab": "ludhiana", "rajasthan": "jaipur", "sikkim": "gangtok", "tamil nadu": "chennai",
    "telangana": "hyderabad", "tripura": "agartala", "uttar pradesh": "lucknow",
    "uttarakhand": "dehradun", "west bengal": "kolkata", "jammu & kashmir": "srinagar",
    "jammu and kashmir": "srinagar", "ladakh": "leh", "andaman & nicobar": "port blair",
}

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def all_city_names():
    return sorted(c["name"] for c in CITIES.values())


# --------------------------------------------------------------------------
# City lookup
# --------------------------------------------------------------------------
def _haversine(la1, lo1, la2, lo2):
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = p2 - p1, math.radians(lo2 - lo1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_coords(text):
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def find_city(query):
    """Return (city_dict, note, suggestions). city_dict is None if not found."""
    q = (query or "").strip().lower()
    if not q:
        return None, None, []

    # 1) latitude, longitude input -> nearest reference city supplies the climate
    coords = _parse_coords(q)
    if coords:
        lat, lon = coords
        if not (6.0 <= lat <= 37.5 and 68.0 <= lon <= 98.0):
            return None, "Coordinates are outside India. Please enter a valid location in India.", []
        ref = min(CITIES.values(), key=lambda c: _haversine(lat, lon, c["lat"], c["lon"]))
        dist = _haversine(lat, lon, ref["lat"], ref["lon"])
        city = dict(ref)
        city.update(name=f"{lat:.2f}°N, {lon:.2f}°E", lat=lat, lon=lon)
        note = f"Using the climate profile of the nearest reference city, {ref['name']} (~{dist:.0f} km away)."
        return city, note, []

    q = ALIASES.get(q, q)
    if q in CITIES:
        return CITIES[q], None, []

    # 2) state name -> state capital / main city
    if q in STATE_CAPITALS:
        city = CITIES[STATE_CAPITALS[q]]
        return city, f"“{query.strip()}” is a state, so its main city {city['name']} is shown.", []

    # 3) prefix / contains match
    starts = [c for k, c in CITIES.items() if k.startswith(q)]
    if len(q) >= 3 and len(starts) >= 1:
        return starts[0], None, []
    contains = [c for k, c in CITIES.items() if q in k]
    if len(q) >= 3 and len(contains) >= 1:
        return contains[0], None, []

    # 4) fuzzy match
    keys = list(CITIES.keys()) + list(ALIASES.keys())
    close = difflib.get_close_matches(q, keys, n=5, cutoff=0.6)
    sugg = []
    for k in close:
        nm = CITIES[ALIASES.get(k, k)]["name"]
        if nm not in sugg:
            sugg.append(nm)
    if sugg and difflib.SequenceMatcher(None, q, sugg[0].lower()).ratio() >= 0.85:
        return CITIES[sugg[0].lower()], None, []
    return None, None, sugg


# --------------------------------------------------------------------------
# Deterministic randomness
# --------------------------------------------------------------------------
def _u(key, ordinal, stream):
    """Deterministic uniform(0,1) from (key, day, stream)."""
    h = hashlib.md5(f"{key}|{ordinal}|{stream}".encode()).digest()
    n = int.from_bytes(h[:8], "big")
    return (n + 0.5) / 2 ** 64


def _norm(key, ordinal, stream):
    return _ND.inv_cdf(min(max(_u(key, ordinal, stream), 1e-6), 1 - 1e-6))


def _ar1(key, ordinal, stream, rho, warmup=25):
    """Auto-correlated N(0,1) series value for a given day (deterministic)."""
    z = _norm(key, ordinal - warmup, stream)
    scale = math.sqrt(1 - rho * rho)
    for d in range(ordinal - warmup + 1, ordinal + 1):
        z = rho * z + scale * _norm(key, d, stream)
    return z


def _interp(arr, month, day):
    """Smooth interpolation of 12 monthly values to a given date (mid-month anchored)."""
    nd = MONTH_DAYS[month - 1]
    pos = (month - 1) + (day - 0.5) / nd - 0.5
    i0 = math.floor(pos)
    f = pos - i0
    return arr[i0 % 12] * (1 - f) + arr[(i0 + 1) % 12] * f


# --------------------------------------------------------------------------
# Astronomy: sunrise / sunset / UV
# --------------------------------------------------------------------------
def _sun_times(lat, lon, date):
    doy = date.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    phi = math.radians(lat)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(phi) * math.cos(decl))
              - math.tan(phi) * math.tan(decl))
    cos_ha = max(-1, min(1, cos_ha))
    ha = math.degrees(math.acos(cos_ha))
    rise = 720 - 4 * (lon + ha) - eqt + 330   # minutes, IST
    sset = 720 - 4 * (lon - ha) - eqt + 330

    def fmt(m):
        m = int(round(m)) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"

    return fmt(rise), fmt(sset), (sset - rise) / 60.0, decl


def _uv_index(lat, decl, cloud, elev):
    zen = abs(math.radians(lat) - decl)
    clear = 13.5 * max(math.cos(zen), 0) ** 2
    uv = clear * (1 - 0.72 * (cloud / 100.0) ** 2) * (1 + 0.06 * elev / 1000.0)
    return max(0, round(uv))


def _uv_label(uv):
    if uv < 3:
        return "Low"
    if uv < 6:
        return "Moderate"
    if uv < 8:
        return "High"
    if uv < 11:
        return "Very High"
    return "Extreme"


def _feels_like(t, rh, wind):
    if t >= 27 and rh >= 40:
        tf = t * 9 / 5 + 32
        hi = (-42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh
              - 0.00683783 * tf * tf - 0.05481717 * rh * rh + 0.00122874 * tf * tf * rh
              + 0.00085282 * tf * rh * rh - 0.00000199 * tf * tf * rh * rh)
        return (hi - 32) * 5 / 9
    if t <= 10 and wind > 4.8:
        return 13.12 + 0.6215 * t - 11.37 * wind ** 0.16 + 0.3965 * t * wind ** 0.16
    return t


# --------------------------------------------------------------------------
# Condition naming + farm advisories
# --------------------------------------------------------------------------
def _condition(rain, cloud, tmax, tmin, hum, month, fog_zone, thunder):
    if rain >= 2.5:
        if thunder and rain >= 12:
            return "Thunderstorm", "fa-cloud-bolt"
        if rain >= 40:
            return "Heavy Rain", "fa-cloud-showers-heavy"
        if rain >= 12:
            return "Moderate Rain", "fa-cloud-showers-heavy"
        return "Light Rain", "fa-cloud-rain"
    if fog_zone and month in (12, 1, 2) and tmin <= 11 and hum >= 80 and cloud < 55:
        return "Foggy Morning, Clear Later", "fa-smog"
    if cloud < 20:
        return ("Hot & Sunny", "fa-sun") if tmax >= 39 else ("Sunny", "fa-sun")
    if cloud < 45:
        return "Mostly Sunny", "fa-cloud-sun"
    if cloud < 70:
        return "Partly Cloudy", "fa-cloud-sun"
    return "Cloudy", "fa-cloud"


def _advisory(rain, tmax, tmin, wind, hum):
    tips = []
    if rain >= 25:
        tips.append("Heavy rain likely: avoid spraying and fertilizer, clear field drainage channels, delay harvesting.")
    elif rain >= 2.5:
        tips.append("Rain likely: postpone irrigation, pesticide spraying and fertilizer top-dressing.")
    if tmax >= 40:
        tips.append("Heat stress risk: irrigate early morning/evening, apply mulch, avoid midday field work.")
    elif tmax >= 35 and rain < 2.5:
        tips.append("Hot and dry: schedule irrigation early morning or late evening to cut evaporation loss.")
    if tmin <= 4:
        tips.append("Frost risk: cover nurseries and young plants, give a light evening irrigation.")
    elif tmin <= 8:
        tips.append("Cold night: protect seedlings and vegetable nurseries.")
    if wind >= 25:
        tips.append("Strong winds: do not spray, support tall crops (maize, banana, sugarcane).")
    if hum >= 85 and 18 <= tmax <= 33:
        tips.append("High humidity: watch for fungal disease (blight, mildew, rust) and scout fields.")
    if not tips:
        tips.append("Good conditions for sowing, weeding, spraying and other field operations.")
    return tips


SEASONS = {
    12: "Winter (Rabi crop growth)", 1: "Winter (Rabi crop growth)", 2: "Winter / Rabi harvest approaching",
    3: "Summer begins (Rabi harvest)", 4: "Summer (Zaid crops, pre-monsoon)", 5: "Peak summer (pre-monsoon)",
    6: "Monsoon onset (Kharif sowing)", 7: "Monsoon (Kharif crops)", 8: "Monsoon (Kharif crops)",
    9: "Late monsoon (Kharif crops)", 10: "Post-monsoon (Kharif harvest, Rabi preparation)",
    11: "Post-monsoon (Rabi sowing)",
}


def _wind_dir(zone, month, u):
    if month in (6, 7, 8, 9):
        base = 5 if zone not in ("northeast", "east_plains") else 4   # SW / S-SE
    elif month in (12, 1, 2):
        base = 1 if zone not in ("north_plains", "gangetic", "arid", "himalaya", "kashmir") else 7
    elif month in (10, 11):
        base = 1
    else:
        base = 6 if zone in ("north_plains", "gangetic", "arid", "gujarat", "central") else 4
    shift = -1 if u < 0.25 else (1 if u > 0.75 else 0)
    return COMPASS[(base + shift) % 8]


# --------------------------------------------------------------------------
# Main forecast
# --------------------------------------------------------------------------
def forecast(city, days=5, include_today=True):
    z = ZONES[city["zone"]]
    lat, lon = city["lat"], city["lon"]
    lapse = (z["ref_elev"] - city["elev"]) * 0.0055
    grid_key = f"{city['zone']}|{math.floor(lat / 2.5)}|{math.floor(lon / 2.5)}"

    start = datetime.now(IST).date()
    first = 0 if include_today else 1
    out = []
    for i in range(first, first + days + (0 if include_today else 0)):
        d = start + timedelta(days=i)
        ordinal = d.toordinal()
        m, dd = d.month, d.day

        n_tmax = _interp(z["tmax"], m, dd) + lapse + city["dt"]
        n_tmin = _interp(z["tmin"], m, dd) + lapse + city["dt"]
        n_rain = _interp(z["rain"], m, dd) * city["rm"]
        n_rdays = max(_interp(z["rdays"], m, dd) * min(city["rm"], 1.6), 0.05)
        n_hum = _interp(z["hum"], m, dd)
        n_wind = _interp(z["wind"], m, dd)

        sigma = z["sigma"][0] if m in (12, 1, 2) else z["sigma"][1]
        zt = _ar1(grid_key, ordinal, "temp", 0.72)
        zw = _ar1(grid_key, ordinal, "wet", 0.62)

        # --- rain occurrence & amount
        p = min(max(n_rdays / MONTH_DAYS[m - 1], 0.01), 0.93)
        thr = _ND.inv_cdf(1 - p)
        rains = zw > thr
        mean_wet = max(n_rain / max(n_rdays, 0.3), 3.0)
        if rains:
            u = min(max(_u(grid_key, ordinal, "amt"), 0.04), 0.97)
            amount = min(max(mean_wet * -math.log(u), 2.5), min(mean_wet * 5, 220))
        else:
            amount = 0.0
        chance = int(round(100 * _ND.cdf((zw - thr) / 0.8)))
        chance = max(3, min(chance, 97))

        # --- cloud
        cloud_base = _ND.cdf(zw)
        wet_weight = 0.35 + 0.65 * math.sqrt(p)
        cloud = 8 + 85 * cloud_base * wet_weight + (14 if rains else 0)
        cloud = max(0, min(100, cloud + 6 * _norm(grid_key, ordinal, "cl")))

        # --- temperature
        tmax = n_tmax + sigma * zt - (3.0 if rains else 0) - 1.5 * (cloud / 100 - 0.4)
        tmin = n_tmin + 0.6 * sigma * zt + 0.9 * (cloud / 100 - 0.4) + 0.5 * _norm(city["name"], ordinal, "tn")
        if tmax - tmin < 3:
            tmax = tmin + 3

        hum = n_hum + 14 * (_ND.cdf(zw) - 0.5) + (8 if rains else 0) + 2 * _norm(city["name"], ordinal, "h")
        hum = max(15, min(98, hum))
        wind = max(2, n_wind * (1 + 0.28 * _norm(grid_key, ordinal, "w")) + (4 if rains and m in (4, 5, 6, 9, 10) else 0))
        wdir = _wind_dir(city["zone"], m, _u(grid_key, ordinal, "wd"))

        thunder = m in (3, 4, 5, 6, 9, 10) and tmax >= 30 and amount >= 12
        cond, icon = _condition(amount, cloud, tmax, tmin, hum, m, z["fog"], thunder)

        rise, sset, dlen, decl = _sun_times(lat, lon, d)
        uv = _uv_index(lat, decl, cloud, city["elev"])
        feels = _feels_like(tmax, hum, wind)
        tips = _advisory(amount, tmax, tmin, wind, hum)
        spray_ok = amount < 2.5 and wind < 15 and chance < 45

        if i == 0:
            label = "Today"
        elif i == 1:
            label = "Tomorrow"
        else:
            label = d.strftime("%A")
        out.append(dict(
            date=d.strftime("%d %b %Y"), short=d.strftime("%d %b"), label=label,
            tmax=round(tmax), tmin=round(tmin), feels=round(feels), humidity=round(hum),
            rain_mm=round(amount, 1), rain_chance=chance, wind=round(wind), wind_dir=wdir,
            cloud=round(cloud), condition=cond, icon=icon, uv=uv, uv_label=_uv_label(uv),
            sunrise=rise, sunset=sset, daylight=f"{int(dlen)}h {int(round((dlen % 1) * 60)):02d}m",
            advisory=tips, spray_ok=spray_ok,
        ))
    return out


def summarize(days_list, month):
    if not days_list:
        return {}
    n = len(days_list)
    total_rain = round(sum(d["rain_mm"] for d in days_list), 1)
    avg_hi = round(sum(d["tmax"] for d in days_list) / n)
    avg_lo = round(sum(d["tmin"] for d in days_list) / n)
    hottest = max(days_list, key=lambda d: d["tmax"])
    coldest = min(days_list, key=lambda d: d["tmin"])
    wettest = max(days_list, key=lambda d: d["rain_mm"])
    rainy_days = sum(1 for d in days_list if d["rain_mm"] >= 2.5)
    spray_days = [d["label"] for d in days_list if d["spray_ok"]]

    if total_rain >= 40:
        irrigation = "No irrigation needed: adequate rain expected. Keep drainage channels open."
    elif total_rain >= 15:
        irrigation = "Reduce irrigation and time it around the rainy days."
    elif avg_hi >= 35:
        irrigation = "High water demand: irrigate in the early morning or evening on dry days."
    else:
        irrigation = "Normal irrigation schedule is fine; check soil moisture before watering."
    return dict(
        avg_hi=avg_hi, avg_lo=avg_lo, total_rain=total_rain, rainy_days=rainy_days,
        hottest=f"{hottest['label']} ({hottest['tmax']}°C)",
        coldest=f"{coldest['label']} ({coldest['tmin']}°C)",
        wettest=(f"{wettest['label']} ({wettest['rain_mm']} mm)" if wettest["rain_mm"] > 0 else "No significant rain"),
        spray_days=spray_days, irrigation=irrigation, season=SEASONS[month],
    )


def get_forecast(query):
    """High-level helper used by the Flask route."""
    city, note, suggestions = find_city(query)
    if not city:
        return dict(ok=False, error=note or f"Sorry, “{query}” is not in the offline city list.",
                    suggestions=suggestions)
    all_days = forecast(city, days=5, include_today=False)
    today = forecast(city, days=1, include_today=True)[0]
    month = datetime.now(IST).month
    return dict(ok=True, city=city, note=note, today=today, days=all_days,
                summary=summarize(all_days, month))

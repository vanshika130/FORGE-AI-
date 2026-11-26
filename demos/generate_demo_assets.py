"""Generate PDF and 200-row CSV demo assets for Demo A (hr_policy) and Demo B (ship_spares)."""

from __future__ import annotations

import csv
from pathlib import Path
import random

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 4))
        for para in body.strip().split("\n\n"):
            story.append(Paragraph(para.strip(), styles["BodyText"]))
            story.append(Spacer(1, 6))
    doc.build(story)


def generate_hr_policy_docs(base_dir: Path) -> None:
    docs = base_dir / "hr_policy" / "docs"
    _write_pdf(
        docs / "leave_policy.pdf",
        "Shore Staff Leave Policy",
        [
            (
                "1. Annual Leave Entitlement",
                "Full-time shore-based staff receive 21 working days of paid annual leave per calendar year after completing the 6-month probation period. During probation, employees accrue 1.5 days per completed month.\n\nUp to 5 unused annual leave days may be carried over into the following calendar year if used before March 31. Unused days beyond 5 are forfeited.",
            ),
            (
                "2. Sick Leave",
                "Shore staff are entitled to 10 paid sick leave days per calendar year. Absence of 3 or more consecutive working days requires a medical certificate submitted to HR within 2 days of returning to work.",
            ),
            (
                "3. Parental and Bereavement Leave",
                "Primary caregivers receive 16 weeks of paid parental leave, and secondary caregivers receive 4 weeks of paid parental leave within 12 months of birth or adoption.\n\nEmployees are granted 5 working days of paid bereavement leave in the event of the death of an immediate family member.",
            ),
        ],
    )
    _write_pdf(
        docs / "travel_expense_policy.pdf",
        "Shore Staff Travel and Expense Policy",
        [
            (
                "1. Flights and Pre-Approval",
                "All business trips estimated above USD 1,500 require written manager pre-approval at least 7 days prior to booking.\n\nEconomy-class airfare is mandatory for all flights under 6 hours. Premium economy may be approved for single-leg flights exceeding 6 hours.",
            ),
            (
                "2. Accommodation and Meal Per Diem",
                "Hotel nightly rate caps are USD 180 for domestic travel and USD 240 for international travel, excluding local taxes.\n\nThe daily meal per diem is USD 65 per full travel day. Alcohol and personal entertainment are not reimbursable.",
            ),
            (
                "3. Expense Reimbursement Deadlines",
                "Itemized receipts must be submitted in the expense portal within 14 calendar days of trip return. Approved claims are reimbursed within 10 business days.",
            ),
        ],
    )
    _write_pdf(
        docs / "code_of_conduct.pdf",
        "Shore Staff Code of Conduct",
        [
            (
                "1. Conflicts of Interest and Gifts",
                "Employees must disclose any financial interest in a vendor or customer to HR Compliance annually.\n\nStaff may not accept gifts, hospitality, or favours from suppliers exceeding USD 50 in value. Cash or cash equivalents of any amount are strictly prohibited.",
            ),
            (
                "2. Confidentiality and Media Inquiries",
                "Company financial records, vessel schedules, and employee records are confidential and must not be shared on public AI tools or personal devices.\n\nAll press or external media inquiries must be routed immediately to Corporate Communications.",
            ),
            (
                "3. Reporting Misconduct",
                "Suspected harassment, safety violations, or policy breaches must be reported to the HR Ethics Hotline or Compliance Officer within 48 hours. Retaliation against good-faith reporters is strictly prohibited.",
            ),
        ],
    )


def generate_ship_spares_docs(base_dir: Path) -> None:
    docs = base_dir / "ship_spares" / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    _write_pdf(
        docs / "supplier_terms_nordic_marine.pdf",
        "Nordic Marine Supply — Commercial Terms & Warranty",
        [
            (
                "1. Delivery and Freight Terms",
                "Nordic Marine Supply delivers under FOB Rotterdam or CIF major European ports. Standard ocean consolidation ships weekly on Tuesdays.\n\nUrgent orders under 50 kg qualify for expedited air freight at a flat surcharge of USD 250.",
            ),
            (
                "2. Warranty and Returns",
                "All OEM parts carry a 12-month warranty from the date of vessel installation or 15 months from shipment, whichever occurs first.\n\nUnused stock returns are accepted within 30 days of delivery subject to a 15% restocking fee.",
            ),
            (
                "3. Payment Terms",
                "Standard invoice terms are Net 30 days from bill of lading date. Late balances accrue 1.5% monthly interest.",
            ),
        ],
    )
    _write_pdf(
        docs / "supplier_terms_pacific_spares.pdf",
        "Pacific Spares Ltd — Commercial Terms & Warranty",
        [
            (
                "1. Delivery and Bulk Discounts",
                "Pacific Spares Ltd ships EXW Singapore or DDP major Asian and Middle East ports.\n\nOrders exceeding USD 5,000 in total merchandise value receive an automatic 5% bulk discount prior to freight.",
            ),
            (
                "2. Warranty and Defective Parts",
                "Propulsion and main engine components carry an 18-month operational warranty. Rubber seals, O-rings, and gaskets carry a 6-month warranty.\n\nDefective parts verified on arrival are replaced free of charge with zero restocking fee.",
            ),
            (
                "3. Payment Terms",
                "Standard payment terms are Net 45 days for approved marine fleet accounts.",
            ),
        ],
    )

    rng = random.Random(2026)
    systems = [
        ("Main Engine", ["Piston Ring Set", "Cylinder Liner", "Fuel Injector Nozzle", "Exhaust Valve Spindle", "Turbocharger Bearing", "Connecting Rod Bolt", "Cylinder Head Gasket", "Camshaft Follower"]),
        ("Auxiliary Generator", ["AVR Voltage Regulator", "Alternator Brush Kit", "Governor Actuator", "Starter Solenoid", "Jacket Water Thermostat", "Lube Oil Filter Element", "Crankcase Breather", "Excitation Diode Bridge"]),
        ("Cooling & Pumps", ["Sea Water Impeller", "Mechanical Shaft Seal", "Plate Heat Exchanger Gasket", "Centrifugal Pump Wear Ring", "Freshwater Cooler Tube", "Ballast Valve Actuator", "Bilge Pump Diaphragm", "Thermostatic Mixing Valve"]),
        ("Navigation & Radar", ["X-Band Magnetron", "Gyrocompass Repeater Card", "AIS Transponder Antenna", "ECDIS Trackball Module", "Echo Sounder Transducer", "Anemometer Wind Cup Set", "VHF Handset Assembly", "GPS Receiver Module"]),
        ("Deck Machinery", ["Mooring Winch Brake Band", "Hydraulic Hose Assembly", "Windlass Clutch Lining", "Cargo Crane Wire Sheave", "Hatch Cover Rubber Packing", "Capstan Gear Pinion", "Fairlead Roller Bush", "Hydraulic Relief Valve"]),
    ]
    suppliers = ["Nordic Marine Supply", "Pacific Spares Ltd"]

    csv_path = docs / "spares_catalogue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "part_number",
                "description",
                "vessel_system",
                "unit_price_usd",
                "supplier",
                "lead_time_days",
                "stock_qty",
            ]
        )
        idx = 1
        for _ in range(5):
            for sys_name, items in systems:
                for item in items:
                    part_no = f"SP-{idx:04d}"
                    variant = f"MK-{(idx % 4) + 1}"
                    desc = f"{item} ({variant})"
                    price = round(rng.uniform(45.0, 2450.0), 2)
                    supplier = suppliers[idx % 2]
                    lead = rng.choice([3, 5, 7, 10, 14, 21, 28])
                    stock = rng.randint(0, 45)
                    writer.writerow([part_no, desc, sys_name, price, supplier, lead, stock])
                    idx += 1


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    generate_hr_policy_docs(base)
    generate_ship_spares_docs(base)
    print("Demo assets generated successfully.")

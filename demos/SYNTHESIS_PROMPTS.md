# Demo Document Synthesis Prompts

This file records the exact prompts used to synthesize the two demonstration document sets (`demos/hr_policy/docs/` and `demos/ship_spares/docs/`).

## Demo A: HR Policy Assistant (`demos/hr_policy/docs/`)
1. **Leave Policy (`leave_policy.pdf`)**:
   > Write a concise, realistic corporate HR Leave Policy for shore-based staff covering: annual leave (21 working days per year after 6 months probation, carry-over cap of 5 days), sick leave (10 paid days per year, medical certificate required after 3 consecutive days), parental leave (16 weeks primary caregiver, 4 weeks secondary caregiver), and bereavement leave (5 working days). Do NOT include payroll tax formulas, individual salary figures, or immigration/visa legal advice.

2. **Travel and Expense Policy (`travel_expense_policy.pdf`)**:
   > Write a realistic Travel and Expense Policy for shore-based staff covering: economy-class flights for trips under 6 hours, hotel nightly caps (\$180 domestic, \$240 international), daily meal per diem (\$65/day), receipt submission deadline (within 14 calendar days of return), and pre-approval threshold (manager approval required above \$1,500).

3. **Code of Conduct (`code_of_conduct.pdf`)**:
   > Write a corporate Code of Conduct for shore-based staff covering: conflict-of-interest disclosure, gift limit (\$50 maximum value), confidentiality of company and vessel data, anti-harassment reporting channels (HR Ethics Hotline within 48 hours), and external media inquiries (must be routed to Corporate Communications).

## Demo B: Ship-Spares Procurement Assistant (`demos/ship_spares/docs/`)
1. **Spares Catalogue (`spares_catalogue.csv`)**:
   > Generate a 200-row marine spare parts CSV with columns: `part_number`, `description`, `vessel_system`, `unit_price_usd`, `supplier`, `lead_time_days`, `stock_qty`. Span 5 vessel systems (`Main Engine`, `Auxiliary Generator`, `Cooling & Pumps`, `Navigation & Radar`, `Deck Machinery`) and 2 suppliers (`Nordic Marine Supply`, `Pacific Spares Ltd`).

2. **Supplier Terms — Nordic Marine Supply (`supplier_terms_nordic_marine.pdf`)**:
   > Write a supplier terms sheet for Nordic Marine Supply covering: FOB Rotterdam / CIF delivery options, standard 12-month OEM parts warranty, 15% restocking fee on unused returns within 30 days, net-30 payment terms, and expedited air-freight surcharge (\$250 flat fee for orders under 50 kg).

3. **Supplier Terms — Pacific Spares Ltd (`supplier_terms_pacific_spares.pdf`)**:
   > Write a supplier terms sheet for Pacific Spares Ltd covering: EXW Singapore / DDP port delivery, 18-month warranty on propulsion and engine spares (6 months on seals/gaskets), zero restocking fee for defective parts, net-45 payment terms, and 5% bulk discount on orders exceeding \$5,000.

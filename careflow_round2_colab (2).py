# CAREFLOW — Round 2 Hospital Operations Intelligence Prototype
# Run this cell in Google Colab. It uses synthetic demo data, not real hospital data.

!pip -q install gradio pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from datetime import datetime

# ----------------------- DEMO DATA -----------------------
DATA = [
    {"Department":"Pediatrics","Visits":657,"Avg wait (min)":60.8,"90th pct (min)":92.0,"Left before care (%)":3.7,"Visits/staff":41},
    {"Department":"Laboratory","Visits":659,"Avg wait (min)":60.6,"90th pct (min)":93.0,"Left before care (%)":3.0,"Visits/staff":24},
    {"Department":"Outpatient","Visits":629,"Avg wait (min)":57.8,"90th pct (min)":90.4,"Left before care (%)":4.8,"Visits/staff":39},
    {"Department":"Neurology","Visits":664,"Avg wait (min)":56.1,"90th pct (min)":88.0,"Left before care (%)":3.0,"Visits/staff":55},
    {"Department":"General Medicine","Visits":666,"Avg wait (min)":55.2,"90th pct (min)":86.0,"Left before care (%)":2.4,"Visits/staff":22},
    {"Department":"Radiology","Visits":658,"Avg wait (min)":54.8,"90th pct (min)":86.0,"Left before care (%)":2.3,"Visits/staff":94},
    {"Department":"Orthopedics","Visits":629,"Avg wait (min)":49.4,"90th pct (min)":79.2,"Left before care (%)":2.1,"Visits/staff":70},
    {"Department":"Emergency","Visits":709,"Avg wait (min)":48.7,"90th pct (min)":79.0,"Left before care (%)":3.7,"Visits/staff":25},
    {"Department":"Surgery","Visits":677,"Avg wait (min)":47.7,"90th pct (min)":79.0,"Left before care (%)":3.1,"Visits/staff":135},
    {"Department":"Cardiology","Visits":663,"Avg wait (min)":44.8,"90th pct (min)":76.8,"Left before care (%)":2.9,"Visits/staff":27},
    {"Department":"Pharmacy","Visits":687,"Avg wait (min)":44.6,"90th pct (min)":74.0,"Left before care (%)":2.9,"Visits/staff":34},
    {"Department":"Admissions","Visits":702,"Avg wait (min)":44.6,"90th pct (min)":76.0,"Left before care (%)":2.6,"Visits/staff":140},
]
df = pd.DataFrame(DATA)

def risk(row):
    if row["Avg wait (min)"] >= 57 or row["Left before care (%)"] >= 4:
        return "High"
    if row["Avg wait (min)"] >= 52 or row["Left before care (%)"] >= 3.2:
        return "Watch"
    return "Stable"

def suggested_action(dept):
    actions = {
        "Pediatrics":"Review arrival clustering and consider short peak-hour intake support.",
        "Laboratory":"Separate sample transport time from processing time; review hand-offs.",
        "Outpatient":"Pilot a peak-hour fast-track/triage lane and compare results after 7 days.",
        "Neurology":"Audit appointment overruns and identify causes of late starts.",
        "General Medicine":"Trial staggered appointment slots to reduce arrival surges.",
        "Radiology":"Check room utilisation and gaps between imaging and report completion.",
        "Orthopedics":"Review room turnover and late appointment starts.",
        "Emergency":"Review triage flow, boarding, and transfer delays with clinical leadership.",
        "Surgery":"Map pre-op to theatre hand-offs and track avoidable idle time.",
        "Cardiology":"Check whether diagnostic scheduling creates downstream bottlenecks.",
        "Pharmacy":"Review prescription queue peaks and repeat-prescription workflow.",
        "Admissions":"Track bed-ready to admission-complete time and clarify hand-off ownership."
    }
    return actions.get(dept, "Review arrival patterns, staffing coverage, and process hand-offs.")

df["Risk"] = df.apply(risk, axis=1)
df["Suggested action"] = df["Department"].map(suggested_action)

def summary():
    total_visits = int(df["Visits"].sum())
    avg_wait = df["Avg wait (min)"].mean()
    avg_left = df["Left before care (%)"].mean()
    return total_visits, avg_wait, avg_left

# ----------------------- CALLBACKS -----------------------
def flow_board(sort_by, department):
    sort_map = {
        "Longest wait": ("Avg wait (min)", False),
        "Most visits": ("Visits", False),
        "Highest walk-out rate": ("Left before care (%)", False),
    }
    col, ascending = sort_map[sort_by]
    shown = df.sort_values(col, ascending=ascending)
    row = df[df["Department"] == department].iloc[0]
    title = f"### {department} snapshot\n**Risk level:** {row['Risk']}\n\n**Suggested action:** {row['Suggested action']}"
    return shown[["Department","Visits","Avg wait (min)","90th pct (min)","Left before care (%)","Visits/staff","Risk"]], title

def forecast(demand_change, capacity_change):
    visits = round(df["Visits"].sum() * (1 + demand_change / 100))
    # Transparent illustrative scenario formula; not a trained predictive model.
    current_wait = df["Avg wait (min)"].mean()
    projected_wait = max(15, current_wait + demand_change * 0.62 - capacity_change * 0.72)
    extra_shifts = max(0, round((projected_wait - 45) / 2.1))
    if capacity_change >= demand_change:
        advice = "In this simplified scenario, capacity growth offsets the demand increase. Validate against actual staffing and arrival data."
    elif demand_change > capacity_change:
        advice = "Demand is rising faster than capacity in this scenario. Consider targeted peak-hour coverage and monitor queue length."
    else:
        advice = "This scenario reduces pressure. Confirm that staffing changes are feasible and safe with hospital operations leaders."
    return f"{visits:,}", f"{projected_wait:.1f} min", f"+{extra_shifts}", advice

def alerts(search_text, reviewed):
    alerts_data = [
        ("Outpatient","Wait exceeds hospital average","Average wait is 57.8 minutes. Review peak-hour flow.","High"),
        ("Pediatrics","Long-tail wait is elevated","90th percentile wait is 92 minutes. Check arrival clustering.","High"),
        ("Laboratory","Diagnostic turnaround needs review","Review sample transport and queue hand-offs.","Watch"),
        ("Emergency","Walk-out rate above 3.5%","Review communication and reassessment intervals with clinical staff.","Watch"),
        ("General Medicine","Morning arrival peak","Explore staggered appointment slots and intake capacity.","Watch"),
    ]
    query = (search_text or "").lower()
    rows = [a for a in alerts_data if query in " ".join(a).lower()]
    if reviewed:
        return "### Alerts reviewed\nAll visible demo alerts have been marked as reviewed."
    if not rows:
        return "No alerts match your search."
    return "\n\n".join([f"**{level} · {dept}: {title}**\n\n{body}" for dept,title,body,level in rows])

def diagnostics(filter_above_target):
    diag = pd.DataFrame([
        {"Service":"Laboratory","Turnaround (min)":78,"Target (min)":60,"Tests":500},
        {"Service":"Radiology","Turnaround (min)":72,"Target (min)":65,"Tests":420},
        {"Service":"Pathology","Turnaround (min)":64,"Target (min)":60,"Tests":210},
        {"Service":"Cardiology tests","Turnaround (min)":54,"Target (min)":60,"Tests":160},
    ])
    if filter_above_target:
        diag = diag[diag["Turnaround (min)"] > diag["Target (min)"]]
    diag["Status"] = diag.apply(lambda r: "Above target" if r["Turnaround (min)"] > r["Target (min)"] else "Within target", axis=1)
    return diag

def investigate(search_text, department):
    query = (search_text or "").lower()
    result = df[df["Department"].str.lower().str.contains(query, na=False)]
    row = df[df["Department"] == department].iloc[0]
    details = (
        f"## Investigation plan: {department}\n"
        f"**Current wait:** {row['Avg wait (min)']} min · **Walk-outs:** {row['Left before care (%)']}% · **Risk:** {row['Risk']}\n\n"
        f"1. Compare hourly arrivals with service-start times.\n"
        f"2. Separate queue time from processing time.\n"
        f"3. Review the highest-delay days and staffing coverage.\n"
        f"4. Pilot this action: {row['Suggested action']}\n"
        f"5. Compare average wait and walk-outs after one week.\n\n"
        "_This is an operational investigation suggestion, not a clinical instruction._"
    )
    return result[["Department","Avg wait (min)","Left before care (%)","Risk"]], details

def export_csv():
    path = "/content/careflow_department_report.csv"
    df.to_csv(path, index=False)
    return path

# ----------------------- DASHBOARD UI -----------------------
total_visits, mean_wait, mean_left = summary()
custom_css = """
.gradio-container {max-width: 1400px !important; background: #f2f7f5 !important;}
#hero {background:#102b36;color:white;padding:24px;border-radius:16px}
#hero h1 {font-size:30px}
.kpi {background:white;border:1px solid #d8e4e0;border-radius:12px;padding:14px}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"), css=custom_css, title="CareFlow | Hospital Operations") as demo:
    gr.Markdown(
        "# 🏥 CareFlow · Hospital Operations Intelligence\n"
        "Patient flow · bottleneck investigation · demand scenarios\n\n"
        "> **Prototype note:** This demo uses synthetic data. It is not connected to a hospital system and is not for clinical decision-making."
    )
    with gr.Row():
        with gr.Column(elem_classes=["kpi"]):
            gr.Markdown("**Visits in demo dataset**")
            gr.Markdown(f"# {total_visits:,}")
        with gr.Column(elem_classes=["kpi"]):
            gr.Markdown("**Mean department wait**")
            gr.Markdown(f"# {mean_wait:.1f} min")
        with gr.Column(elem_classes=["kpi"]):
            gr.Markdown("**Mean walk-out rate**")
            gr.Markdown(f"# {mean_left:.2f}%")
        with gr.Column(elem_classes=["kpi"]):
            gr.Markdown("**Departments analysed**")
            gr.Markdown(f"# {len(df)}")

    with gr.Tabs():
        with gr.Tab("Flow Board"):
            gr.Markdown("## Department flow board\nSort departments and select one to inspect its risk and suggested next step.")
            with gr.Row():
                sort_by = gr.Dropdown(["Longest wait","Most visits","Highest walk-out rate"], value="Longest wait", label="Sort departments")
                dept = gr.Dropdown(df["Department"].tolist(), value="Outpatient", label="Department to inspect")
            flow_table = gr.Dataframe(value=df[["Department","Visits","Avg wait (min)","90th pct (min)","Left before care (%)","Visits/staff","Risk"]], interactive=False, label="Department metrics")
            dept_details = gr.Markdown()
            sort_by.change(flow_board, [sort_by, dept], [flow_table, dept_details])
            dept.change(flow_board, [sort_by, dept], [flow_table, dept_details])
            flow_table.select(lambda evt: evt, None, None, show_progress="hidden")
            _, initial_details = flow_board("Longest wait","Outpatient")
            dept_details.value = initial_details

        with gr.Tab("Forecast"):
            gr.Markdown("## Demand and capacity scenario\nMove the sliders to see how a simple illustrative scenario changes estimated visits and waiting time. This is not a trained AI model.")
            with gr.Row():
                demand = gr.Slider(-10, 20, value=4, step=1, label="Demand change (%)")
                capacity = gr.Slider(-10, 20, value=0, step=1, label="Staff capacity change (%)")
            with gr.Row():
                forecast_visits = gr.Textbox(label="Estimated visits next period", interactive=False)
                forecast_wait = gr.Textbox(label="Projected average wait", interactive=False)
                forecast_staff = gr.Textbox(label="Estimated extra shifts (rough)", interactive=False)
            forecast_advice = gr.Markdown()
            for control in [demand, capacity]:
                control.change(forecast, [demand, capacity], [forecast_visits, forecast_wait, forecast_staff, forecast_advice])
            initial_forecast = forecast(4,0)
            forecast_visits.value, forecast_wait.value, forecast_staff.value, forecast_advice.value = initial_forecast

        with gr.Tab("Alerts"):
            gr.Markdown("## Operational alerts\nThese are sample alerts generated from the demo dataset.")
            alert_search = gr.Textbox(label="Search alerts", placeholder="e.g. outpatient, laboratory…")
            reviewed = gr.Checkbox(label="Mark all alerts reviewed", value=False)
            alert_output = gr.Markdown()
            alert_search.change(alerts, [alert_search, reviewed], alert_output)
            reviewed.change(alerts, [alert_search, reviewed], alert_output)
            alert_output.value = alerts("", False)

        with gr.Tab("Diagnostics"):
            gr.Markdown("## Diagnostic turnaround\nCompare illustrative turnaround time with target values.")
            above_target = gr.Checkbox(label="Show only services above target", value=False)
            diag_table = gr.Dataframe(value=diagnostics(False), interactive=False)
            above_target.change(diagnostics, above_target, diag_table)
            gr.Markdown("**Suggested experiment:** Track collection-to-lab receipt separately from processing time to find where delays occur.")

        with gr.Tab("Investigate"):
            gr.Markdown("## Investigate a department\nSearch the department table and select a department for an evidence-led investigation plan.")
            with gr.Row():
                search = gr.Textbox(label="Search departments", placeholder="Type a department name…")
                investigate_dept = gr.Dropdown(df["Department"].tolist(), value="Outpatient", label="Department")
            investigation_table = gr.Dataframe(value=df[["Department","Avg wait (min)","Left before care (%)","Risk"]], interactive=False)
            investigation_detail = gr.Markdown()
            search.change(investigate, [search, investigate_dept], [investigation_table, investigation_detail])
            investigate_dept.change(investigate, [search, investigate_dept], [investigation_table, investigation_detail])
            _, initial_investigation = investigate("", "Outpatient")
            investigation_detail.value = initial_investigation

        with gr.Tab("Export"):
            gr.Markdown("## Export the demo data\nDownload a CSV of the department metrics, risk labels, and suggested actions.")
            export_btn = gr.Button("Generate CSV report", variant="primary")
            export_file = gr.File(label="Download report")
            export_btn.click(export_csv, outputs=export_file)

    gr.Markdown("---\n**CareFlow Round 2 prototype** · Built with Python + Gradio · Synthetic demonstration data · Validate all assumptions before real-world use.")

demo.launch(share=True, debug=True)

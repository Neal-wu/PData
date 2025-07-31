from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Optional
import datetime
from personal_analytics import PersonalAnalytics, LifeCategory, CategoryReport

# Global analytics instance - will be set by main app
analytics = None

def set_analytics_instance(analytics_instance):
    """Set the analytics instance from the main app."""
    global analytics
    analytics = analytics_instance

# Create router for personal analytics endpoints
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])

@analytics_router.get("/", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """Main analytics dashboard page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Personal Analytics Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .content {
                padding: 30px;
            }
            .category-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .category-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                border-left: 5px solid #667eea;
                transition: transform 0.3s ease;
            }
            .category-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .category-card h3 {
                margin: 0 0 15px 0;
                color: #333;
                font-size: 1.3em;
            }
            .category-card p {
                margin: 0 0 15px 0;
                color: #666;
                line-height: 1.5;
            }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            .stats-section {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            .stat-item {
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .stat-item h4 {
                margin: 0 0 5px 0;
                color: #667eea;
                font-size: 1.5em;
            }
            .stat-item p {
                margin: 0;
                color: #666;
                font-size: 0.9em;
            }
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                color: white;
                text-decoration: none;
                font-weight: 500;
            }
            .back-link:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <a href="/" class="back-link">← Back to Main</a>
                <h1>📊 Personal Analytics Dashboard</h1>
                <p>Analyze your life across different categories</p>
            </div>
            
            <div class="content">
                <div class="stats-section">
                    <h2>📈 Overall Statistics</h2>
                    <div class="stats-grid" id="overall-stats">
                        <div class="stat-item">
                            <h4 id="total-activities">-</h4>
                            <p>Total Activities</p>
                        </div>
                        <div class="stat-item">
                            <h4 id="categories-tracked">-</h4>
                            <p>Categories Tracked</p>
                        </div>
                        <div class="stat-item">
                            <h4 id="most-active-category">-</h4>
                            <p>Most Active Category</p>
                        </div>
                        <div class="stat-item">
                            <h4 id="weekly-reports">-</h4>
                            <p>Weekly Reports</p>
                        </div>
                    </div>
                </div>
                
                <h2>🎯 Life Categories</h2>
                <div class="category-grid">
                    <div class="category-card">
                        <h3>📚💼 Education & Work</h3>
                        <p>Track your learning activities, study sessions, courses, work projects, and career progress.</p>
                        <a href="/analytics/education_work" class="btn">View Education & Work Report</a>
                    </div>
                    
                    <div class="category-card">
                        <h3>🏥 Healthcare</h3>
                        <p>Monitor your health activities, medications, exercise, sleep, and wellness patterns.</p>
                        <a href="/analytics/healthcare" class="btn">View Healthcare Report</a>
                    </div>
                    
                    <div class="category-card">
                        <h3>🎮 Entertainment</h3>
                        <p>Analyze your entertainment activities, media consumption, hobbies, and leisure time.</p>
                        <a href="/analytics/entertainment" class="btn">View Entertainment Report</a>
                    </div>
                    
                    <div class="category-card">
                        <h3>💰 Finance</h3>
                        <p>Track your financial activities, spending, investments, banking, and budget management.</p>
                        <a href="/analytics/finance" class="btn">View Finance Report</a>
                    </div>
                    
                    <div class="category-card">
                        <h3>📋 Other Activities</h3>
                        <p>View miscellaneous activities and other life areas not covered by main categories.</p>
                        <a href="/analytics/other" class="btn">View Other Report</a>
                    </div>
                    
                    <div class="category-card">
                        <h3>📊 All Categories</h3>
                        <p>Get a comprehensive view of all your activities across all life categories.</p>
                        <a href="/analytics/all" class="btn">View All Reports</a>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Load overall statistics
            async function loadOverallStats() {
                try {
                    const response = await fetch('/analytics/stats');
                    const stats = await response.json();
                    
                    document.getElementById('total-activities').textContent = stats.total_activities || 0;
                    document.getElementById('categories-tracked').textContent = stats.categories_tracked || 0;
                    document.getElementById('most-active-category').textContent = stats.most_active_category || 'None';
                    document.getElementById('weekly-reports').textContent = stats.weekly_reports_generated || 0;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }
            
            // Load data when page loads
            loadOverallStats();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@analytics_router.get("/education_work", response_class=HTMLResponse)
async def education_work_report():
    """Generate and display education and work report."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        report = analytics.generate_category_report(LifeCategory.EDUCATION_WORK)
        return _generate_category_report_html(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating education and work report: {str(e)}")

@analytics_router.get("/healthcare", response_class=HTMLResponse)
async def healthcare_report():
    """Generate and display healthcare report."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        report = analytics.generate_category_report(LifeCategory.HEALTHCARE)
        return _generate_category_report_html(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating healthcare report: {str(e)}")

@analytics_router.get("/entertainment", response_class=HTMLResponse)
async def entertainment_report():
    """Generate and display entertainment report."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        report = analytics.generate_category_report(LifeCategory.ENTERTAINMENT)
        return _generate_category_report_html(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating entertainment report: {str(e)}")

@analytics_router.get("/finance", response_class=HTMLResponse)
async def finance_report():
    """Generate and display finance report."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        report = analytics.generate_category_report(LifeCategory.FINANCE)
        return _generate_category_report_html(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating finance report: {str(e)}")

@analytics_router.get("/other", response_class=HTMLResponse)
async def other_report():
    """Generate and display other activities report."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        report = analytics.generate_category_report(LifeCategory.OTHER)
        return _generate_category_report_html(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating other report: {str(e)}")

@analytics_router.get("/all", response_class=HTMLResponse)
async def all_reports():
    """Generate and display all category reports."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        reports = analytics.generate_all_category_reports()
        return _generate_all_reports_html(reports)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating all reports: {str(e)}")

@analytics_router.get("/stats")
async def get_analytics_statistics():
    """Get overall analytics statistics."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        distribution = analytics.get_category_distribution()
        total_activities = sum(distribution.values())
        categories_tracked = len([c for c, count in distribution.items() if count > 0])
        most_active_category = max(distribution.items(), key=lambda x: x[1]) if distribution else ("None", 0)
        most_active_category_display = f"{most_active_category[0].value} ({most_active_category[1]})" if most_active_category[0] != "None" else "None"
        
        return JSONResponse(content={
            "total_activities": total_activities,
            "categories_tracked": categories_tracked,
            "most_active_category": most_active_category_display,
            "weekly_reports_generated": analytics.get_generated_reports_count(),  # Actual number of reports generated
            "category_distribution": {k.value: v for k, v in distribution.items()}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@analytics_router.get("/search/{category}")
async def search_category_activities(category: str, query: str):
    """Search for activities within a specific category."""
    if not analytics:
        raise HTTPException(status_code=500, detail="Analytics not initialized")
    
    try:
        category_enum = LifeCategory(category.lower())
        results = analytics.search_category_activities(category_enum, query)
        
        return JSONResponse(content={
            "category": category,
            "query": query,
            "results": [
                {"subject": t[0], "predicate": t[1], "object": t[2]}
                for t in results
            ],
            "count": len(results)
        })
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching activities: {str(e)}")

def _generate_category_report_html(report: CategoryReport) -> HTMLResponse:
    """Generate HTML for a category report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report.title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .insight {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }}
            .recommendation {{ background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .statistics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
            .stat-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
            .activity-item {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .back-btn {{ 
                display: inline-block; 
                padding: 10px 20px; 
                background-color: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <a href="/analytics" class="back-btn">← Back to Analytics</a>
        
        <div class="header">
            <h1>{report.title}</h1>
            <p>Generated on: {report.generated_at.strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Period: {report.period_start.strftime('%B %d, %Y')} to {report.period_end.strftime('%B %d, %Y')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <p>{report.summary}</p>
        </div>
        
        <div class="section">
            <h2>Statistics</h2>
            <div class="statistics">
                {_generate_statistics_html(report.statistics)}
            </div>
        </div>
        
        <div class="section">
            <h2>Insights ({len(report.insights)})</h2>
            {_generate_insights_html(report.insights)}
        </div>
        
        <div class="section">
            <h2>Top Activities</h2>
            {_generate_activities_html(report.top_activities)}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {_generate_recommendations_html(report.recommendations)}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def _generate_all_reports_html(reports: dict) -> HTMLResponse:
    """Generate HTML for all category reports."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>All Category Reports</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .category-section { margin: 30px 0; padding: 20px; border: 2px solid #ddd; border-radius: 10px; }
            .category-title { color: #333; font-size: 1.5em; margin-bottom: 15px; }
            .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
            .insight { background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 3px; }
            .recommendation { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }
            .back-btn { 
                display: inline-block; 
                padding: 10px 20px; 
                background-color: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <a href="/analytics" class="back-btn">← Back to Analytics</a>
        
        <div class="header">
            <h1>All Category Reports</h1>
            <p>Generated on: """ + datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p') + """</p>
        </div>
    """
    
    for category, report in reports.items():
        html_content += f"""
        <div class="category-section">
            <h2 class="category-title">{category.value.title()}</h2>
            <div class="summary">
                <p><strong>Summary:</strong> {report.summary}</p>
                <p><strong>Activities:</strong> {report.statistics.get('total_activities', 0)}</p>
            </div>
            
            <h3>Insights:</h3>
            {_generate_insights_html(report.insights)}
            
            <h3>Recommendations:</h3>
            {_generate_recommendations_html(report.recommendations)}
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def _generate_statistics_html(statistics: dict) -> str:
    """Generate HTML for statistics section."""
    if "error" in statistics:
        return f'<div class="stat-card"><h3>Error</h3><p>{statistics["error"]}</p></div>'
    
    html_parts = []
    for key, value in statistics.items():
        if isinstance(value, (int, float)):
            html_parts.append(f'<div class="stat-card"><h3>{value}</h3><p>{key.replace("_", " ").title()}</p></div>')
        elif isinstance(value, tuple):
            html_parts.append(f'<div class="stat-card"><h3>{value[1]}</h3><p>{value[0]}</p></div>')
        else:
            html_parts.append(f'<div class="stat-card"><h3>{str(value)}</h3><p>{key.replace("_", " ").title()}</p></div>')
    
    return "".join(html_parts)

def _generate_insights_html(insights: list) -> str:
    """Generate HTML for insights section."""
    if not insights:
        return "<p>No insights available for this period.</p>"
    
    html_parts = []
    for insight in insights:
        html_parts.append(f"""
            <div class="insight">
                <h3>{insight.title}</h3>
                <p>{insight.description}</p>
                <small>Confidence: {insight.confidence:.1%}</small>
            </div>
        """)
    
    return "".join(html_parts)

def _generate_activities_html(activities: list) -> str:
    """Generate HTML for activities section."""
    if not activities:
        return "<p>No activities recorded for this period.</p>"
    
    html_parts = []
    for i, (activity, count) in enumerate(activities, 1):
        html_parts.append(f'<div class="activity-item"><strong>{i}.</strong> {activity} ({count} times)</div>')
    
    return "".join(html_parts)

def _generate_recommendations_html(recommendations: list) -> str:
    """Generate HTML for recommendations section."""
    if not recommendations:
        return "<p>No recommendations available.</p>"
    
    html_parts = []
    for recommendation in recommendations:
        html_parts.append(f'<div class="recommendation">• {recommendation}</div>')
    
    return "".join(html_parts)

def integrate_analytics_to_main_app(app):
    """Integrate the analytics router into the main FastAPI app."""
    app.include_router(analytics_router) 
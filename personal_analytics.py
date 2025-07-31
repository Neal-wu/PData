import networkx as nx
import json
import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import re
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class LifeCategory(Enum):
    EDUCATION_WORK = "education_work"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    FINANCE = "finance"
    OTHER = "other"

@dataclass
class CategoryInsight:
    category: LifeCategory
    title: str
    description: str
    confidence: float
    data: Dict[str, Any]
    timestamp: datetime.datetime

@dataclass
class CategoryReport:
    category: LifeCategory
    title: str
    generated_at: datetime.datetime
    period_start: datetime.datetime
    period_end: datetime.datetime
    summary: str
    insights: List[CategoryInsight]
    statistics: Dict[str, Any]
    recommendations: List[str]
    top_activities: List[Tuple[str, int]]
    trends: Dict[str, Any]

class PersonalAnalytics:
    """
    A comprehensive analytics system for personal knowledge graphs
    that categorizes data into life areas and generates insights.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.triples_history = []
        self.category_keywords = self._initialize_category_keywords()
        self.generated_reports = []  # Track generated reports
        
    def _initialize_category_keywords(self) -> Dict[LifeCategory, List[str]]:
        """Initialize keywords for each life category."""
        return {
            LifeCategory.EDUCATION_WORK: [
                'study', 'learn', 'course', 'class', 'lecture', 'homework', 'assignment',
                'exam', 'test', 'quiz', 'book', 'read', 'research', 'project', 'presentation',
                'degree', 'university', 'college', 'school', 'professor', 'teacher', 'student',
                'knowledge', 'skill', 'training', 'workshop', 'seminar', 'tutorial', 'lesson',
                'python', 'programming', 'coding', 'algorithm', 'data structure', 'math',
                'science', 'history', 'literature', 'language', 'grammar', 'vocabulary',
                'work', 'job', 'career', 'office', 'meeting', 'presentation', 'deadline',
                'client', 'customer', 'colleague', 'coworker', 'boss', 'manager', 'team',
                'project', 'task', 'assignment', 'report', 'analysis', 'strategy',
                'business', 'company', 'organization', 'department', 'position', 'role',
                'salary', 'promotion', 'interview', 'resume', 'application', 'hire'
            ],
            LifeCategory.HEALTHCARE: [
                'health', 'medical', 'doctor', 'hospital', 'clinic', 'medicine', 'medication',
                'pill', 'tablet', 'capsule', 'ibuprofen', 'acetaminophen', 'aspirin',
                'symptom', 'pain', 'headache', 'fever', 'cough', 'cold', 'flu', 'sick',
                'exercise', 'workout', 'gym', 'run', 'walk', 'jog', 'swim', 'bike',
                'sleep', 'rest', 'nap', 'insomnia', 'fatigue', 'energy', 'tired',
                'diet', 'food', 'eat', 'meal', 'breakfast', 'lunch', 'dinner', 'snack',
                'water', 'drink', 'hydration', 'vitamin', 'supplement', 'protein',
                'blood pressure', 'heart rate', 'temperature', 'weight', 'height',
                'stress', 'anxiety', 'depression', 'mood', 'mental health', 'therapy',
                'meditation', 'yoga', 'breathing', 'relaxation'
            ],
            LifeCategory.ENTERTAINMENT: [
                'movie', 'film', 'watch', 'tv', 'television', 'show', 'series', 'episode',
                'game', 'play', 'video game', 'board game', 'card game', 'puzzle',
                'music', 'song', 'album', 'artist', 'band', 'concert', 'performance',
                'book', 'novel', 'story', 'fiction', 'non-fiction', 'magazine', 'article',
                'party', 'celebration', 'birthday', 'holiday', 'vacation', 'travel',
                'hobby', 'craft', 'art', 'drawing', 'painting', 'photography', 'cooking',
                'dance', 'karaoke', 'comedy', 'joke', 'fun', 'enjoy', 'entertain',
                'social media', 'facebook', 'instagram', 'twitter', 'youtube', 'netflix'
            ],
            LifeCategory.FINANCE: [
                'money', 'finance', 'financial', 'bank', 'banking', 'account', 'savings',
                'checking', 'credit', 'debit', 'card', 'cash', 'payment', 'bill',
                'expense', 'income', 'salary', 'wage', 'earn', 'spend', 'buy', 'purchase',
                'investment', 'stock', 'bond', 'fund', 'portfolio', 'retirement',
                'insurance', 'loan', 'mortgage', 'debt', 'budget', 'budgeting',
                'tax', 'taxes', 'tax return', 'refund', 'deduction', 'expense',
                'shopping', 'store', 'market', 'grocery', 'retail', 'online',
                'subscription', 'membership', 'fee', 'charge', 'transaction',
                'deposit', 'withdrawal', 'transfer', 'balance', 'statement'
            ]
        }
    
    def load_graph_from_triples(self, triples: List[Tuple[str, str, str]], 
                               timestamp: Optional[datetime.datetime] = None,
                               activity_date: Optional[datetime.date] = None):
        """Load knowledge graph from extracted triples with activity date."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        if activity_date is None:
            activity_date = datetime.datetime.now().date()
            
        self.triples_history.append({
            'triples': triples,
            'timestamp': timestamp,
            'activity_date': activity_date,
            'count': len(triples)
        })
        
        for subject, predicate, obj in triples:
            self.graph.add_edge(subject, obj, predicate=predicate, 
                              timestamp=timestamp, activity_date=activity_date)
            
        print(f"Loaded {len(triples)} triples into personal analytics for {activity_date}")
    
    def categorize_triple(self, triple: Tuple[str, str, str]) -> LifeCategory:
        """Categorize a triple into a life category."""
        subject, predicate, obj = triple
        text_to_check = f"{subject} {predicate} {obj}".lower()
        
        # Check each category
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_to_check:
                    return category
        
        return LifeCategory.OTHER
    
    def get_triples_by_category(self, category: LifeCategory, 
                               start_date: Optional[datetime.date] = None,
                               end_date: Optional[datetime.date] = None) -> List[Tuple[str, str, str, datetime.date]]:
        """Get triples for a specific category within a date range."""
        if start_date is None:
            start_date = datetime.datetime.now().date() - datetime.timedelta(days=7)
        if end_date is None:
            end_date = datetime.datetime.now().date()
        
        category_triples = []
        for history_item in self.triples_history:
            activity_date = history_item.get('activity_date', history_item['timestamp'].date())
            # Ensure we're comparing dates with dates
            if isinstance(activity_date, datetime.datetime):
                activity_date = activity_date.date()
            if start_date <= activity_date <= end_date:
                for triple in history_item['triples']:
                    if self.categorize_triple(triple) == category:
                        category_triples.append((*triple, activity_date))
        
        return category_triples
    
    def generate_category_report(self, category: LifeCategory, 
                               week_start: Optional[datetime.date] = None) -> CategoryReport:
        """Generate a weekly report for a specific category."""
        if week_start is None:
            today = datetime.datetime.now().date()
            week_start = today - datetime.timedelta(days=today.weekday())
        
        end_date = week_start + datetime.timedelta(days=6)
        
        category_triples_with_dates = self.get_triples_by_category(category, week_start, end_date)
        category_triples = [(t[0], t[1], t[2]) for t in category_triples_with_dates]  # Remove date for compatibility
        
        if not category_triples:
            start_time = datetime.datetime.combine(week_start, datetime.time.min)
            end_time = datetime.datetime.combine(end_date, datetime.time.max)
            return self._create_empty_category_report(category, start_time, end_time)
        
        # Generate insights
        insights = self._generate_category_insights(category, category_triples)
        
        # Generate recommendations
        recommendations = self._generate_category_recommendations(category, insights, category_triples)
        
        # Create summary
        summary = self._create_category_summary(category, category_triples, insights)
        
        # Get statistics
        stats = self._get_category_statistics(category, category_triples)
        
        # Get top activities
        top_activities = self._get_top_category_activities(category_triples)
        
        # Get trends
        trends = self._get_category_trends(category, category_triples)
        
        # Create datetime objects for the report period
        start_time = datetime.datetime.combine(week_start, datetime.time.min)
        end_time = datetime.datetime.combine(end_date, datetime.time.max)
        
        report = CategoryReport(
            category=category,
            title=f"Weekly {category.value.title()} Report - Week of {week_start.strftime('%B %d, %Y')}",
            generated_at=datetime.datetime.now(),
            period_start=start_time,
            period_end=end_time,
            summary=summary,
            insights=insights,
            statistics=stats,
            recommendations=recommendations,
            top_activities=top_activities,
            trends=trends
        )
        
        # Track this generated report
        self.generated_reports.append({
            'category': category,
            'generated_at': report.generated_at,
            'period_start': report.period_start,
            'period_end': report.period_end
        })
        
        return report
    
    def _generate_category_insights(self, category: LifeCategory, 
                                  triples: List[Tuple[str, str, str]]) -> List[CategoryInsight]:
        """Generate insights specific to a category."""
        insights = []
        
        if category == LifeCategory.EDUCATION_WORK:
            insights.extend(self._generate_education_work_insights(triples))
        elif category == LifeCategory.HEALTHCARE:
            insights.extend(self._generate_healthcare_insights(triples))
        elif category == LifeCategory.ENTERTAINMENT:
            insights.extend(self._generate_entertainment_insights(triples))
        elif category == LifeCategory.FINANCE:
            insights.extend(self._generate_finance_insights(triples))
        else:
            insights.extend(self._generate_general_insights(triples))
        
        return insights
    
    def _generate_education_work_insights(self, triples: List[Tuple[str, str, str]]) -> List[CategoryInsight]:
        """Generate education and work-specific insights."""
        insights = []
        
        # Study time analysis
        study_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                    for word in ['study', 'learn', 'read', 'homework'])]
        if study_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.EDUCATION_WORK,
                title="Study Activities",
                description=f"Engaged in {len(study_activities)} study-related activities this week",
                confidence=0.8,
                data={"study_activities": len(study_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        # Work activities analysis
        work_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                   for word in ['work', 'job', 'meeting', 'project', 'task', 'client'])]
        if work_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.EDUCATION_WORK,
                title="Work Activities",
                description=f"Engaged in {len(work_activities)} work-related activities this week",
                confidence=0.8,
                data={"work_activities": len(work_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        # Subject analysis
        subjects = [t[2] for t in triples if any(word in t[2].lower() 
                                               for word in ['python', 'math', 'science', 'history', 'english', 'project', 'analysis'])]
        if subjects:
            unique_subjects = list(set(subjects))
            insights.append(CategoryInsight(
                category=LifeCategory.EDUCATION_WORK,
                title="Subjects/Projects Covered",
                description=f"Worked on {len(unique_subjects)} different subjects/projects: {', '.join(unique_subjects)}",
                confidence=0.7,
                data={"subjects": unique_subjects},
                timestamp=datetime.datetime.now()
            ))
        
        return insights
    
    def _generate_healthcare_insights(self, triples: List[Tuple[str, str, str]]) -> List[CategoryInsight]:
        """Generate healthcare-specific insights."""
        insights = []
        
        # Medication tracking
        medications = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                               for word in ['took', 'medication', 'pill', 'ibuprofen', 'acetaminophen'])]
        if medications:
            insights.append(CategoryInsight(
                category=LifeCategory.HEALTHCARE,
                title="Medication Usage",
                description=f"Took medication {len(medications)} times this week",
                confidence=0.9,
                data={"medication_count": len(medications)},
                timestamp=datetime.datetime.now()
            ))
        
        # Exercise tracking
        exercise_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                       for word in ['exercise', 'workout', 'run', 'walk', 'gym'])]
        if exercise_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.HEALTHCARE,
                title="Exercise Activities",
                description=f"Engaged in {len(exercise_activities)} exercise activities this week",
                confidence=0.8,
                data={"exercise_count": len(exercise_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        # Sleep tracking
        sleep_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                    for word in ['sleep', 'slept', 'rest'])]
        if sleep_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.HEALTHCARE,
                title="Sleep Tracking",
                description=f"Recorded {len(sleep_activities)} sleep-related activities this week",
                confidence=0.7,
                data={"sleep_activities": len(sleep_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        return insights
    
    def _generate_entertainment_insights(self, triples: List[Tuple[str, str, str]]) -> List[CategoryInsight]:
        """Generate entertainment-specific insights."""
        insights = []
        
        # Media consumption
        media_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                    for word in ['watch', 'movie', 'tv', 'show', 'game', 'play'])]
        if media_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.ENTERTAINMENT,
                title="Media Consumption",
                description=f"Engaged in {len(media_activities)} entertainment activities this week",
                confidence=0.8,
                data={"media_activities": len(media_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        # Social activities
        social_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                     for word in ['party', 'celebration', 'social', 'meet'])]
        if social_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.ENTERTAINMENT,
                title="Social Activities",
                description=f"Participated in {len(social_activities)} social activities this week",
                confidence=0.7,
                data={"social_activities": len(social_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        return insights
    
    def _generate_finance_insights(self, triples: List[Tuple[str, str, str]]) -> List[CategoryInsight]:
        """Generate finance-specific insights."""
        insights = []
        
        # Spending analysis
        spending_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                       for word in ['spend', 'buy', 'purchase', 'payment', 'expense'])]
        if spending_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.FINANCE,
                title="Spending Activities",
                description=f"Recorded {len(spending_activities)} spending-related activities this week",
                confidence=0.8,
                data={"spending_activities": len(spending_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        # Banking activities
        banking_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                      for word in ['bank', 'account', 'deposit', 'withdrawal', 'transfer'])]
        if banking_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.FINANCE,
                title="Banking Activities",
                description=f"Engaged in {len(banking_activities)} banking-related activities this week",
                confidence=0.7,
                data={"banking_activities": len(banking_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        # Investment activities
        investment_activities = [t for t in triples if any(word in f"{t[0]} {t[1]} {t[2]}".lower() 
                                                         for word in ['investment', 'stock', 'bond', 'fund', 'portfolio'])]
        if investment_activities:
            insights.append(CategoryInsight(
                category=LifeCategory.FINANCE,
                title="Investment Activities",
                description=f"Engaged in {len(investment_activities)} investment-related activities this week",
                confidence=0.7,
                data={"investment_activities": len(investment_activities)},
                timestamp=datetime.datetime.now()
            ))
        
        return insights
    
    def _generate_general_insights(self, triples: List[Tuple[str, str, str]]) -> List[CategoryInsight]:
        """Generate general insights for uncategorized activities."""
        insights = []
        
        if triples:
            insights.append(CategoryInsight(
                category=LifeCategory.OTHER,
                title="Other Activities",
                description=f"Recorded {len(triples)} other activities this week",
                confidence=0.6,
                data={"other_activities": len(triples)},
                timestamp=datetime.datetime.now()
            ))
        
        return insights
    
    def _generate_category_recommendations(self, category: LifeCategory, 
                                         insights: List[CategoryInsight], 
                                         triples: List[Tuple[str, str, str]]) -> List[str]:
        """Generate recommendations for a category."""
        recommendations = []
        
        if category == LifeCategory.EDUCATION_WORK:
            if len(triples) < 5:
                recommendations.append("Consider dedicating more time to learning and work activities")
            if not any('exercise' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Balance work/study time with physical activities")
            if not any('meeting' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Consider tracking work meetings and collaborations")
        
        elif category == LifeCategory.HEALTHCARE:
            if len(triples) < 3:
                recommendations.append("Try to track more health-related activities")
            if not any('exercise' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Consider adding regular exercise to your routine")
            if not any('sleep' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Monitor your sleep patterns for better health")
        
        elif category == LifeCategory.ENTERTAINMENT:
            if len(triples) < 2:
                recommendations.append("Make time for entertainment and relaxation")
            if not any('social' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Consider more social activities for better well-being")
        
        elif category == LifeCategory.FINANCE:
            if len(triples) < 3:
                recommendations.append("Start tracking your financial activities regularly")
            if not any('budget' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Consider creating and tracking a budget")
            if not any('investment' in f"{t[0]} {t[1]} {t[2]}".lower() for t in triples):
                recommendations.append("Consider tracking investment activities for better financial planning")
        
        if not recommendations:
            recommendations.append(f"Great job maintaining balance in your {category.value} activities!")
        
        return recommendations
    
    def _create_category_summary(self, category: LifeCategory, 
                               triples: List[Tuple[str, str, str]], 
                               insights: List[CategoryInsight]) -> str:
        """Create a summary for a category."""
        if not triples:
            return f"No {category.value} activities were recorded this week."
        
        summary_parts = [
            f"This week, you recorded {len(triples)} {category.value}-related activities.",
            f"Key insights: {len(insights)} patterns and trends were identified."
        ]
        
        for insight in insights[:2]:  # Top 2 insights
            summary_parts.append(f"• {insight.description}")
        
        return " ".join(summary_parts)
    
    def _get_category_statistics(self, category: LifeCategory, 
                               triples: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Get statistics for a category."""
        if not triples:
            return {"error": f"No {category.value} data for this week"}
        
        entities = [triple[0] for triple in triples] + [triple[2] for triple in triples]
        relationships = [triple[1] for triple in triples]
        
        return {
            "total_activities": len(triples),
            "unique_entities": len(set(entities)),
            "unique_relationships": len(set(relationships)),
            "most_frequent_activity": Counter(entities).most_common(1)[0] if entities else None,
            "most_frequent_relationship": Counter(relationships).most_common(1)[0] if relationships else None
        }
    
    def _get_top_category_activities(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, int]]:
        """Get top activities for a category."""
        if not triples:
            return []
        
        activities = []
        for triple in triples:
            activity = f"{triple[0]} {triple[1]} {triple[2]}"
            activities.append(activity)
        
        return Counter(activities).most_common(5)
    
    def _get_category_trends(self, category: LifeCategory, 
                           triples: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Get trends for a category."""
        if not triples:
            return {"error": "No data for trend analysis"}
        
        # Simple trend analysis
        daily_counts = defaultdict(int)
        for history_item in self.triples_history:
            for triple in history_item['triples']:
                if self.categorize_triple(triple) == category:
                    day = history_item['timestamp'].date()
                    daily_counts[day] += 1
        
        return {
            "daily_distribution": dict(daily_counts),
            "most_active_day": max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None,
            "average_activities_per_day": sum(daily_counts.values()) / len(daily_counts) if daily_counts else 0
        }
    
    def _create_empty_category_report(self, category: LifeCategory, 
                                    start_time: datetime.datetime, 
                                    end_time: datetime.datetime) -> CategoryReport:
        """Create an empty report when no data is available."""
        return CategoryReport(
            category=category,
            title=f"Weekly {category.value.title()} Report",
            generated_at=datetime.datetime.now(),
            period_start=start_time,
            period_end=end_time,
            summary=f"No {category.value} activities were recorded this week.",
            insights=[],
            statistics={"error": f"No {category.value} data available"},
            recommendations=[f"Start tracking your {category.value} activities"],
            top_activities=[],
            trends={"error": "No data for trend analysis"}
        )
    
    def generate_all_category_reports(self, week_start: Optional[datetime.date] = None) -> Dict[LifeCategory, CategoryReport]:
        """Generate reports for all categories."""
        reports = {}
        for category in LifeCategory:
            reports[category] = self.generate_category_report(category, week_start)
        return reports
    
    def export_category_report_to_json(self, report: CategoryReport) -> str:
        """Export a category report to JSON format."""
        # Convert trends to JSON-serializable format
        trends_serializable = {}
        for key, value in report.trends.items():
            if key == "daily_distribution":
                trends_serializable[key] = {str(k): v for k, v in value.items()}
            elif key == "most_active_day" and value is not None:
                trends_serializable[key] = (str(value[0]), value[1])
            else:
                trends_serializable[key] = value
        
        report_dict = {
            "category": report.category.value,
            "title": report.title,
            "generated_at": report.generated_at.isoformat(),
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "summary": report.summary,
            "insights": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "data": insight.data,
                    "timestamp": insight.timestamp.isoformat()
                }
                for insight in report.insights
            ],
            "statistics": report.statistics,
            "recommendations": report.recommendations,
            "top_activities": report.top_activities,
            "trends": trends_serializable
        }
        return json.dumps(report_dict, indent=2)
    
    def get_category_distribution(self) -> Dict[LifeCategory, int]:
        """Get the distribution of activities across categories."""
        distribution = defaultdict(int)
        for history_item in self.triples_history:
            for triple in history_item['triples']:
                category = self.categorize_triple(triple)
                distribution[category] += 1
        return dict(distribution)
    
    def search_category_activities(self, category: LifeCategory, query: str) -> List[Tuple[str, str, str]]:
        """Search for activities within a specific category."""
        category_triples = []
        for history_item in self.triples_history:
            for triple in history_item['triples']:
                if self.categorize_triple(triple) == category:
                    text = f"{triple[0]} {triple[1]} {triple[2]}".lower()
                    if query.lower() in text:
                        category_triples.append(triple)
        return category_triples
    
    def get_generated_reports_count(self) -> int:
        """Get the total number of reports that have been generated."""
        return len(self.generated_reports)
    
    def get_recent_reports(self, days: int = 7) -> List[dict]:
        """Get reports generated in the last N days."""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_reports = [
            report for report in self.generated_reports 
            if report['generated_at'] >= cutoff_date
        ]
        return recent_reports

# Example usage
# if __name__ == "__main__":
    # Create analytics instance
    # analytics = PersonalAnalytics()
    
    # Sample triples for testing
    # sample_triples = [
        # ("I", "studied", "Python programming"),
        # ("I", "took", "400mg ibuprofen"),
        # ("I", "watched", "a movie"),
        # ("I", "met", "my friend"),
        # ("I", "exercised", "30 minutes"),
        # ("I", "read", "a book"),
        # ("I", "attended", "a meeting"),
        # ("I", "slept", "8 hours"),
        # ("I", "played", "video games"),
        # ("I", "called", "my family")
    # ]
    
    # Load triples
    # analytics.load_graph_from_triples(sample_triples)
    
    # Generate reports for all categories
    # reports = analytics.generate_all_category_reports()
    
    # for category, report in reports.items():
        # print(f"\n=== {category.value.upper()} REPORT ===")
        # print(f"Title: {report.title}")
        # print(f"Summary: {report.summary}")
        # print(f"Activities: {report.statistics.get('total_activities', 0)}")
        # print(f"Insights: {len(report.insights)}")
        # print(f"Recommendations: {len(report.recommendations)}")
        
        # for insight in report.insights:
            # print(f"  - {insight.title}: {insight.description}")
        
        # for rec in report.recommendations:
            # print(f"  - {rec}")
    
    # Get category distribution
    # distribution = analytics.get_category_distribution()
    # print(f"\n=== CATEGORY DISTRIBUTION ===")
    # for category, count in distribution.items():
        # print(f"{category.value}: {count} activities") 
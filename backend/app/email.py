from mailjet_rest import Client
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Mailjet client
mailjet = Client(auth=(settings.MAILJET_API_KEY, settings.MAILJET_API_SECRET), version='v3.1')

def send_email(to_email: str, to_name: str, subject: str, html_content: str, text_content: str = None):
    """
    Send email using Mailjet
    
    Args:
        to_email: Recipient email address
        to_name: Recipient name
        subject: Email subject
        html_content: HTML email body
        text_content: Plain text email body (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data = {
            'Messages': [
                {
                    "From": {
                        "Email": settings.MAILJET_FROM_EMAIL,
                        "Name": settings.MAILJET_FROM_NAME
                    },
                    "To": [
                        {
                            "Email": to_email,
                            "Name": to_name
                        }
                    ],
                    "Subject": subject,
                    "HTMLPart": html_content,
                }
            ]
        }
        
        # Add text part if provided
        if text_content:
            data['Messages'][0]['TextPart'] = text_content
        
        result = mailjet.send.create(data=data)
        print("Result;", result.status_code, result.json())
        
        if result.status_code == 200:
            logger.info(f"Email sent successfully to {to_email}")
            return True
        else:
            logger.error(f"Failed to send email to {to_email}: {result.status_code} - {result.json()}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending email to {to_email}: {str(e)}")
        return False

def send_model_completed_email(to_email: str, to_name: str, run_id: int, model_type: str, optimal_clusters: int = None):
    """
    Send notification when model training is completed
    """
    subject = f"Model Training Completed - Run #{run_id}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #3498db; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
            .info-box {{ background: white; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #3498db; }}
            .button {{ display: inline-block; padding: 12px 24px; background: #3498db; color: white; text-decoration: none; border-radius: 6px; margin-top: 15px; }}
            .footer {{ text-align: center; margin-top: 20px; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Training Completed</h1>
            </div>
            <div class="content">
                <p>Hello {to_name},</p>
                
                <p>Your model training has completed successfully!</p>
                
                <div class="info-box">
                    <strong>Run Details:</strong><br>
                    Run ID: #{run_id}<br>
                    Model Type: {model_type.replace('_', ' ').title()}<br>
                    {f'Optimal Clusters: {optimal_clusters}' if optimal_clusters else ''}
                </div>
                
                <p>You can now view your results, including plots and metrics.</p>
                
                <a href="{settings.FRONTEND_URL}/runs/{run_id}" class="button">View Results</a>
                
                <p style="margin-top: 20px;">If you have any questions, please don't hesitate to reach out.</p>
                
                <div class="footer">
                    <p>This is an automated message from Model Training Platform</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    Model Training Completed
    
    Hello {to_name},
    
    Your model training has completed successfully!
    
    Run Details:
    - Run ID: #{run_id}
    - Model Type: {model_type.replace('_', ' ').title()}
    {f'- Optimal Clusters: {optimal_clusters}' if optimal_clusters else ''}
    
    View your results at: {settings.FRONTEND_URL}/runs/{run_id}
    
    This is an automated message from Model Training Platform
    """
    
    return send_email(to_email, to_name, subject, html_content, text_content)

def send_clinician_review_email(to_email: str, to_name: str, run_id: int, data_scientist_name: str):
    """
    Send notification to clinician when results are ready for review
    """
    to_email = "nathaniayeo@gmail.com"
    print(to_email)
    subject = f"Model Results Ready for Review - Run #{run_id}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #2c3e50; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #5b7ed8; color: white; padding: 30px 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .content {{ background: #f0f4ff; padding: 30px; border-radius: 0 0 8px 8px; }}
            .info-box {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 5px solid #5b7ed8; }}
            .info-box strong {{ color: #5b7ed8; }}
            .button {{ display: inline-block; padding: 14px 28px; background: #5b7ed8; color: white; text-decoration: none; border-radius: 6px; margin-top: 15px; }}
            .footer {{ text-align: center; margin-top: 20px; color: #8b9dc3; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Results Ready for Review</h1>
            </div>
            <div class="content">
                <p>Hello Dr. {to_name},</p>
                
                <p>A new model training has been completed and is ready for your clinical review.</p>
                
                <div class="info-box">
                    <strong>Request Details:</strong><br>
                    Run ID: #{run_id}<br>
                    Requested by: {data_scientist_name}<br>
                    Status: Awaiting your feedback
                </div>
                
                <p>Please review the results and provide your clinical feedback when convenient.</p>
                
                <a href="{settings.FRONTEND_URL}/runs/{run_id}" class="button" style="color: white;">Review Results</a>
                
                <p style="margin-top: 20px;">Your expertise is valuable in validating these findings.</p>
                
                <div class="footer">
                    <p>This is an automated message from Model Training Platform</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    Results Ready for Review
    
    Hello Dr. {to_name},
    
    A new model training has been completed and is ready for your clinical review.
    
    Request Details:
    - Run ID: #{run_id}
    - Requested by: {data_scientist_name}
    - Status: Awaiting your feedback
    
    Please review the results at: {settings.FRONTEND_URL}/runs/{run_id}
    
    Your expertise is valuable in validating these findings.
    
    This is an automated message from Model Training Platform
    """
    
    return send_email(to_email, to_name, subject, html_content, text_content)

def send_feedback_added_email(to_email: str, to_name: str, run_id: int, clinician_name: str):
    """
    Send notification to data scientist when clinician adds feedback
    """
    subject = f"Feedback Added to Run #{run_id}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #27ae60; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #d4edda; padding: 30px; border-radius: 0 0 8px 8px; }}
            .info-box {{ background: white; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #27ae60; }}
            .button {{ display: inline-block; padding: 12px 24px; background: #27ae60; color: white; text-decoration: none; border-radius: 6px; margin-top: 15px; }}
            .footer {{ text-align: center; margin-top: 20px; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>New Feedback Received</h1>
            </div>
            <div class="content">
                <p>Hello {to_name},</p>
                
                <p>Clinical feedback has been added to your model run.</p>
                
                <div class="info-box">
                    <strong>Feedback Details:</strong><br>
                    Run ID: #{run_id}<br>
                    Feedback by: {clinician_name}<br>
                    Status: Ready to review
                </div>
                
                <p>The clinician has completed their review and added their feedback.</p>
                
                <a href="{settings.FRONTEND_URL}/runs/{run_id}" class="button">View Feedback</a>
                
                <p style="margin-top: 20px;">Review the feedback to improve your model training process.</p>
                
                <div class="footer">
                    <p>This is an automated message from Model Training Platform</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    New Feedback Received
    
    Hello {to_name},
    
    Clinical feedback has been added to your model run.
    
    Feedback Details:
    - Run ID: #{run_id}
    - Feedback by: {clinician_name}
    - Status: Ready to review
    
    View the feedback at: {settings.FRONTEND_URL}/runs/{run_id}
    
    Review the feedback to improve your model training process.
    
    This is an automated message from Model Training Platform
    """
    
    return send_email(to_email, to_name, subject, html_content, text_content)

def send_model_failed_email(to_email: str, to_name: str, run_id: int, model_type: str):
    """
    Send notification when model training fails
    """
    subject = f"Model Training Failed - Run #{run_id}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #e74c3c; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f8d7da; padding: 30px; border-radius: 0 0 8px 8px; }}
            .info-box {{ background: white; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #e74c3c; }}
            .button {{ display: inline-block; padding: 12px 24px; background: #e74c3c; color: white; text-decoration: none; border-radius: 6px; margin-top: 15px; }}
            .footer {{ text-align: center; margin-top: 20px; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Training Failed</h1>
            </div>
            <div class="content">
                <p>Hello {to_name},</p>
                
                <p>Unfortunately, your model training encountered an error and could not be completed.</p>
                
                <div class="info-box">
                    <strong>Run Details:</strong><br>
                    Run ID: #{run_id}<br>
                    Model Type: {model_type.replace('_', ' ').title()}<br>
                    Status: Failed
                </div>
                
                <p>Please check your dataset and try again. If the problem persists, contact support.</p>
                
                <a href="{settings.FRONTEND_URL}/runs/{run_id}" class="button">View Details</a>
                
                <div class="footer">
                    <p>This is an automated message from Model Training Platform</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    Model Training Failed
    
    Hello {to_name},
    
    Unfortunately, your model training encountered an error and could not be completed.
    
    Run Details:
    - Run ID: #{run_id}
    - Model Type: {model_type.replace('_', ' ').title()}
    - Status: Failed
    
    Please check your dataset and try again. If the problem persists, contact support.
    
    View details at: {settings.FRONTEND_URL}/runs/{run_id}
    
    This is an automated message from Model Training Platform
    """
    
    return send_email(to_email, to_name, subject, html_content, text_content)
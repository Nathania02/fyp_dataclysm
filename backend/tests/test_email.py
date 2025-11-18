"""
Unit tests for email sending functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.send_email import (
    send_email,
    send_model_completed_email,
    send_clinician_review_email,
    send_model_failed_email
)


class TestSendEmail:
    """Test base email sending function"""
    
    @patch('app.send_email.mailjet')
    def test_send_email_success(self, mock_mailjet):
        """Test successful email sending"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_mailjet.send.create.return_value = mock_response
        
        result = send_email(
            to_email="test@example.com",
            to_name="Test User",
            subject="Test Subject",
            html_content="<p>Test HTML</p>",
            text_content="Test Text"
        )
        
        assert result is True
        mock_mailjet.send.create.assert_called_once()
    
    @patch('app.send_email.mailjet')
    def test_send_email_without_text_content(self, mock_mailjet):
        """Test email sending without text content"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_mailjet.send.create.return_value = mock_response
        
        result = send_email(
            to_email="test@example.com",
            to_name="Test User",
            subject="Test Subject",
            html_content="<p>Test HTML</p>"
        )
        
        assert result is True
    
    @patch('app.send_email.mailjet')
    def test_send_email_failure(self, mock_mailjet):
        """Test failed email sending"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_mailjet.send.create.return_value = mock_response
        
        result = send_email(
            to_email="test@example.com",
            to_name="Test User",
            subject="Test Subject",
            html_content="<p>Test HTML</p>"
        )
        
        assert result is False
    
    @patch('app.send_email.mailjet')
    def test_send_email_exception(self, mock_mailjet):
        """Test email sending with exception"""
        mock_mailjet.send.create.side_effect = Exception("Connection error")
        
        result = send_email(
            to_email="test@example.com",
            to_name="Test User",
            subject="Test Subject",
            html_content="<p>Test HTML</p>"
        )
        
        assert result is False
    
    @patch('app.send_email.mailjet')
    def test_send_email_data_structure(self, mock_mailjet):
        """Test that email data structure is correct"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_mailjet.send.create.return_value = mock_response
        
        send_email(
            to_email="test@example.com",
            to_name="Test User",
            subject="Test Subject",
            html_content="<p>Test HTML</p>",
            text_content="Test Text"
        )
        
        call_args = mock_mailjet.send.create.call_args
        data = call_args.kwargs['data']
        
        assert 'Messages' in data
        assert len(data['Messages']) == 1
        assert data['Messages'][0]['To'][0]['Email'] == "test@example.com"
        assert data['Messages'][0]['Subject'] == "Test Subject"
        assert 'TextPart' in data['Messages'][0]


class TestModelCompletedEmail:
    """Test model completion email"""
    
    @patch('app.send_email.send_email')
    def test_send_model_completed_email_with_clusters(self, mock_send_email):
        """Test sending completion email with optimal clusters"""
        mock_send_email.return_value = True
        
        result = send_model_completed_email(
            to_email="scientist@example.com",
            to_name="Scientist",
            run_id=123,
            model_type="kmeans",
            optimal_clusters=3
        )
        
        assert result is True
        mock_send_email.assert_called_once()
        
        # Check that the email contains the run information
        call_args = mock_send_email.call_args
        assert "Run #{run_id}".replace("{run_id}", "123") in call_args[0][2]  # subject
        assert "123" in call_args[0][3]  # html content
        assert "3" in call_args[0][3]  # optimal clusters in html
    
    @patch('app.send_email.send_email')
    def test_send_model_completed_email_without_clusters(self, mock_send_email):
        """Test sending completion email without optimal clusters"""
        mock_send_email.return_value = True
        
        result = send_model_completed_email(
            to_email="scientist@example.com",
            to_name="Scientist",
            run_id=456,
            model_type="lca",
            optimal_clusters=None
        )
        
        assert result is True
        mock_send_email.assert_called_once()
    
    @patch('app.send_email.send_email')
    def test_model_completed_email_model_type_formatting(self, mock_send_email):
        """Test that model type is properly formatted in email"""
        mock_send_email.return_value = True
        
        send_model_completed_email(
            to_email="test@example.com",
            to_name="Test",
            run_id=1,
            model_type="kmeans_dtw",
            optimal_clusters=4
        )
        
        call_args = mock_send_email.call_args
        html_content = call_args[0][3]
        text_content = call_args[0][4]
        
        # Check that underscores are replaced with spaces and title cased
        assert "Kmeans Dtw" in html_content
        assert "Kmeans Dtw" in text_content


class TestClinicianReviewEmail:
    """Test clinician review email"""
    
    @patch('app.send_email.send_email')
    def test_send_clinician_review_email(self, mock_send_email):
        """Test sending clinician review email"""
        mock_send_email.return_value = True
        
        result = send_clinician_review_email(
            to_email="clinician@example.com",
            to_name="Doctor Smith",
            run_id=789,
            data_scientist_name="John Doe"
        )
        
        assert result is True
        mock_send_email.assert_called_once()
        
        call_args = mock_send_email.call_args
        assert "789" in call_args[0][3]  # run_id in html
        assert "John Doe" in call_args[0][3]  # scientist name in html
    
    @patch('app.send_email.send_email')
    def test_clinician_review_email_content(self, mock_send_email):
        """Test clinician review email contains correct content"""
        mock_send_email.return_value = True
        
        send_clinician_review_email(
            to_email="clinician@example.com",
            to_name="Doctor Smith",
            run_id=100,
            data_scientist_name="Jane Doe"
        )
        
        call_args = mock_send_email.call_args
        html_content = call_args[0][3]
        text_content = call_args[0][4]
        
        assert "Ready for Review" in html_content
        assert "clinical review" in html_content
        assert "100" in text_content
        assert "Jane Doe" in text_content

class TestModelFailedEmail:
    """Test model failed email"""
    
    @patch('app.send_email.send_email')
    def test_send_model_failed_email(self, mock_send_email):
        """Test sending model failed email"""
        mock_send_email.return_value = True
        
        result = send_model_failed_email(
            to_email="scientist@example.com",
            to_name="Scientist",
            run_id=999,
            model_type="gbtm"
        )
        
        assert result is True
        mock_send_email.assert_called_once()
        
        call_args = mock_send_email.call_args
        assert "999" in call_args[0][3]
        assert "Failed" in call_args[0][2]  # subject
    
    @patch('app.send_email.send_email')
    def test_model_failed_email_content(self, mock_send_email):
        """Test model failed email contains error information"""
        mock_send_email.return_value = True
        
        send_model_failed_email(
            to_email="test@example.com",
            to_name="Test User",
            run_id=111,
            model_type="kmeans"
        )
        
        call_args = mock_send_email.call_args
        html_content = call_args[0][3]
        text_content = call_args[0][4]
        
        assert "Failed" in html_content
        assert "error" in html_content
        assert "Failed" in text_content
        assert "111" in text_content


class TestEmailIntegration:
    """Test email integration scenarios"""
    
    @patch('app.send_email.mailjet')
    def test_all_email_types_callable(self, mock_mailjet):
        """Test that all email functions are callable without errors"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_mailjet.send.create.return_value = mock_response
        
        # Test all email functions
        functions = [
            (send_model_completed_email, ["test@example.com", "Test", 1, "kmeans", 3]),
            (send_clinician_review_email, ["test@example.com", "Test", 1, "Scientist"]),
            (send_model_failed_email, ["test@example.com", "Test", 1, "kmeans"])
        ]
        
        for func, args in functions:
            result = func(*args)
            assert result is True
    
    @patch('app.send_email.mailjet')
    def test_email_with_special_characters(self, mock_mailjet):
        """Test email with special characters in content"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_mailjet.send.create.return_value = mock_response
        
        result = send_email(
            to_email="test@example.com",
            to_name="Test <User>",
            subject="Test & Special Characters",
            html_content="<p>Content with &, <, >, \"</p>"
        )
        
        assert result is True

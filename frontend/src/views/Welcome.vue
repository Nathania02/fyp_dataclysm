<template>
  <div class="welcome-page">
    <LandingHeader />

    <!-- Hero Section with Background Image -->
    <section class="hero-section">
      <div class="hero-overlay"></div>
      <div class="hero-content">
        <h1 class="hero-title">
          Welcome to <span class="highlight">Model Training Platform</span>
        </h1>
        <p class="hero-subtitle">
          Collaborative Model Training Platform for Sepsis Phenotyping
        </p>
        <p class="hero-description">
          Empowering data scientists and clinicians to work together in developing 
          and validating machine learning models for better patient outcomes.
        </p>
        <div class="hero-actions">
          <button @click="$router.push('/signup')" class="btn btn-primary btn-lg">
            Get Started
          </button>
          <button @click="scrollToSection('features')" class="btn btn-outline btn-lg">
            Learn More
          </button>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features-section" id="features">
      <div class="container">
        <div class="section-header">
          <h2>What do we offer?</h2>
          <p>Everything you need for collaborative model development</p>
        </div>
        
        <div class="card-grid">
          <div class="card feature-card">
            <div class="feature-icon">
                <img src="../assets/train-model.png" alt="Train Model Icon" width="100"/>
            </div>
            <h3>Model Training Hub</h3>
            <p>
              Upload datasets and configure clustering algorithms including K-Means, 
              K-Means with DTW, and Latent Class Analysis.
            </p>
          </div>

          <div class="card feature-card">
            <div class="feature-icon">
                <img src="../assets/collaboration.png" alt="Collaborate Icon" width="100"/>
            </div>
            <h3>Collaboration with Clinicians</h3>
            <p>
              Bridge the gap between data scientists and clinicians with 
              seamless feedback and annotation workflows.
            </p>
          </div>

          <!-- <div class="card feature-card">
            <div class="feature-icon">🔄</div>
            <h3>Compare Results</h3>
            <p>
              Side-by-side comparison of multiple model runs with 
              comprehensive visualizations and metrics.
            </p>
          </div> -->

          <div class="card feature-card">
            <div class="feature-icon">
                <img src="../assets/real-time.png" alt="real time Icon" width="100"/>
            </div>
            <h3>Real-time Monitoring</h3>
            <p>
              Track model training progress in real-time with instant 
              notifications and status updates.
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- How It Works Section -->
    <section class="how-it-works-section">
      <div class="container">
        <div class="section-header">
          <h2>How It Works (In Progress)</h2>
          <p>Simple workflow for powerful results</p>
        </div>

        <div class="steps-container">
          <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-content">
              <h3>Upload & Configure</h3>
              <p>
                Data scientists upload DuckDB datasets and configure model parameters 
                using YAML configuration files.
              </p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-content">
              <h3>Train Models</h3>
              <p>
                Run clustering algorithms (K-Means, DTW, LCA) on your data with 
                automatic optimal cluster detection.
              </p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-content">
              <h3>Clinical Review</h3>
              <p>
                Send results to clinicians for validation and feedback on the 
                discovered phenotypes.
              </p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">4</div>
            <div class="step-content">
              <h3>Iterate & Improve</h3>
              <p>
                Incorporate clinical feedback and iterate on models to achieve 
                better patient stratification.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <LandingFooter />
  </div>
</template>

<script setup>
import LandingHeader from '../components/Header.vue'
import LandingFooter from '../components/Footer.vue'

const scrollToSection = (sectionId) => {
  const element = document.getElementById(sectionId)
  if (element) {
    const navHeight = 70
    const elementPosition = element.getBoundingClientRect().top + window.pageYOffset
    const offsetPosition = elementPosition - navHeight
    
    window.scrollTo({
      top: offsetPosition,
      behavior: 'smooth'
    })
  }
}
</script>

<style scoped>
.welcome-page {
  min-height: 100vh;
  background: #f5f7fa;
}

.btn-lg {
  padding: 1rem 2.5rem;
  font-size: 1.1rem;
  font-weight: 600;
}

.btn-outline {
  background: transparent;
  color: white;
  border: 2px solid white;
}

.btn-outline:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: white;
}

/* Hero Section */
.hero-section {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-image: url('../assets/welcome-hero3.jpg');
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  margin-top: 70px;
}

.hero-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(43, 81, 93, 0.92) 0%, rgba(43, 81, 93, 0.85) 100%);
}

.hero-content {
  position: relative;
  z-index: 1;
  max-width: 1200px;
  margin: 0 auto;
  padding: 4rem 2rem;
  text-align: center;
  color: white;
}

.hero-title {
  font-size: 4rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  line-height: 1.2;
}

.highlight {
  color: #5dade2;
  text-shadow: 0 0 30px rgba(93, 173, 226, 0.5);
}

.hero-subtitle {
  font-size: 1.8rem;
  margin-bottom: 1rem;
  font-weight: 500;
  opacity: 0.95;
}

.hero-description {
  font-size: 1.2rem;
  margin-bottom: 2.5rem;
  opacity: 0.9;
  line-height: 1.7;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

.hero-actions {
  display: flex;
  gap: 1.5rem;
  justify-content: center;
}

/* Features Section */
.features-section {
  padding: 5rem 0;
  background: #f8f9fa;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.section-header {
  text-align: center;
  margin-bottom: 3rem;
}

.section-header h2 {
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 1rem;
}

.section-header p {
  font-size: 1.2rem;
  color: #7f8c8d;
}

.feature-card {
  text-align: center;
  transition: transform 0.3s, box-shadow 0.3s;
  height: 100%;
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 24px rgba(43, 81, 93, 0.15);
}

.feature-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.feature-card h3 {
  font-size: 1.3rem;
  color: #2c3e50;
  margin-bottom: 0.75rem;
}

.feature-card p {
  color: #7f8c8d;
  line-height: 1.6;
}

/* How It Works Section */
.how-it-works-section {
  background: white;
  padding: 5rem 0;
}

.steps-container {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.step-card {
  display: flex;
  gap: 2rem;
  align-items: start;
  padding: 2rem;
  border-left: 4px solid #2B515D;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.2s, box-shadow 0.2s;
}

.step-card:hover {
  transform: translateX(8px);
  box-shadow: 0 4px 12px rgba(43, 81, 93, 0.15);
}

.step-number {
  width: 60px;
  height: 60px;
  background: #2B515D;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.8rem;
  font-weight: 700;
  flex-shrink: 0;
}

.step-content h3 {
  font-size: 1.5rem;
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.step-content p {
  color: #7f8c8d;
  line-height: 1.6;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero-title {
    font-size: 2.5rem;
  }

  .hero-subtitle {
    font-size: 1.3rem;
  }

  .hero-actions {
    flex-direction: column;
  }

  .step-card {
    flex-direction: column;
    text-align: center;
  }

  .step-number {
    margin: 0 auto;
  }
}
</style>
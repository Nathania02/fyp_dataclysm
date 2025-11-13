<template>
  <div class="about-page">
    <LandingHeader />
    
    <!-- About Us Section -->
    <section class="about-section">
      <div class="container">
        <div class="section-header">
          <h2>About Us</h2>
          <p class="about-description">
            We are students from Singapore Management University developing this tool 
            for our Final Year Project in collaboration with SingHealth.
          </p>
        </div>
        
        <div class="team-grid">
          <div class="team-member">
            <div class="member-photo">
              <div class="photo-placeholder">NY</div>
            </div>
            <h3>Nathania Yeo</h3>
            <p class="role">Data Scientist / Sprint Master</p>
          </div>

          <div class="team-member">
            <div class="member-photo">
              <div class="photo-placeholder">OWX</div>
            </div>
            <h3>Ooi Wen Xian</h3>
            <p class="role">Data Scientist / Project Manager</p>
          </div>

          <div class="team-member">
            <div class="member-photo">
              <div class="photo-placeholder">JYL</div>
            </div>
            <h3>Jin Yun Lin</h3>
            <p class="role">Data Scientist</p>
          </div>

          <div class="team-member">
            <div class="member-photo">
              <div class="photo-placeholder">S</div>
            </div>
            <h3>Sriya</h3>
            <p class="role">Data Engineer / Data Scientist</p>
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
</script>

<style scoped>
.about-page {
  min-height: 100vh;
  background: #f5f7fa;
}

/* About Us Section */
.about-section {
  padding: 8rem 0 5rem 0; /* Extra top padding for fixed header */
  background: white;
  min-height: calc(100vh - 300px); /* Ensure decent height */
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
  max-width: 700px;
  margin: 0 auto;
}

.about-description {
  line-height: 1.8;
}

.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 3rem;
  max-width: 1000px;
  margin: 0 auto;
}

.team-member {
  text-align: center;
}

.member-photo {
  width: 150px;
  height: 150px;
  margin: 0 auto 1.5rem;
  border-radius: 50%;
  overflow: hidden;
  border: 4px solid #2B515D;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s, box-shadow 0.3s;
}

.member-photo:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 20px rgba(43, 81, 93, 0.3);
}

.photo-placeholder {
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #2B515D 0%, #3d6b7a 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
  font-weight: 700;
  color: white;
}

.team-member h3 {
  font-size: 1.3rem;
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.team-member .role {
  color: #7f8c8d;
  font-size: 0.95rem;
  line-height: 1.5;
}

/* Responsive Design */
@media (max-width: 768px) {
  .about-section {
    padding: 6rem 0 3rem 0;
  }
  
  .team-grid {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 2rem;
  }

  .member-photo {
    width: 120px;
    height: 120px;
  }
  
  .section-header h2 {
    font-size: 2rem;
  }
}
</style>
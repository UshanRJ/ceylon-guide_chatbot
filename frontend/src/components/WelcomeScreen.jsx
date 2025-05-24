// frontend/src/components/WelcomeScreen.jsx
import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FiArrowRight, FiMapPin, FiCloud, FiDollarSign, FiGlobe } from 'react-icons/fi';

const WelcomeContainer = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
`;

const BackgroundElements = styled.div`
  position: absolute;
  width: 100%;
  height: 100%;
  overflow: hidden;
`;

const FloatingElement = styled(motion.div)`
  position: absolute;
  font-size: ${props => props.size || '3rem'};
  opacity: 0.1;
`;

const ContentWrapper = styled.div`
  max-width: 800px;
  padding: 2rem;
  text-align: center;
  position: relative;
  z-index: 1;
`;

const WelcomeTitle = styled(motion.h1)`
  color: white;
  font-size: 3.5rem;
  font-weight: 700;
  font-family: 'Poppins', sans-serif;
  margin-bottom: 1rem;
  
  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const WelcomeSubtitle = styled(motion.p)`
  color: rgba(255, 255, 255, 0.9);
  font-size: 1.5rem;
  font-weight: 400;
  margin-bottom: 3rem;
  line-height: 1.6;
  
  @media (max-width: 768px) {
    font-size: 1.2rem;
  }
`;

const FeaturesGrid = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
`;

const FeatureCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 2rem;
  border-radius: 16px;
  color: white;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-5px);
  }
`;

const FeatureIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 1rem;
  display: flex;
  justify-content: center;
`;

const FeatureTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
`;

const FeatureDescription = styled.p`
  font-size: 0.95rem;
  opacity: 0.9;
  line-height: 1.5;
`;

const GetStartedButton = styled(motion.button)`
  background: white;
  color: #667eea;
  border: none;
  padding: 1rem 3rem;
  font-size: 1.1rem;
  font-weight: 600;
  border-radius: 50px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const ProgressIndicator = styled.div`
  position: absolute;
  bottom: 2rem;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 0.5rem;
`;

const ProgressDot = styled(motion.div)`
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: ${props => props.active ? 'white' : 'rgba(255, 255, 255, 0.3)'};
  cursor: pointer;
  transition: all 0.3s ease;
`;

const WelcomeScreen = ({ onComplete }) => {
    const [currentStep, setCurrentStep] = useState(0);

    const features = [
        {
            icon: <FiMapPin />,
            title: "Explore Places",
            description: "Discover hidden gems and popular destinations across Sri Lanka"
        },
        {
            icon: <FiCloud />,
            title: "Weather Updates",
            description: "Get real-time weather information for any location"
        },
        {
            icon: <FiDollarSign />,
            title: "Currency Converter",
            description: "Convert between LKR and major currencies instantly"
        },
        {
            icon: <FiGlobe />,
            title: "Language Support",
            description: "Translate between English, Sinhala, and Tamil"
        }
    ];

    const floatingElements = [
        { emoji: "🌴", top: "10%", left: "5%", size: "4rem" },
        { emoji: "🏖️", top: "20%", right: "10%", size: "3rem" },
        { emoji: "🐘", bottom: "30%", left: "8%", size: "5rem" },
        { emoji: "🏛️", top: "15%", left: "70%", size: "3.5rem" },
        { emoji: "🌊", bottom: "20%", right: "5%", size: "4rem" },
        { emoji: "☕", top: "50%", right: "15%", size: "3rem" },
        { emoji: "🛺", bottom: "10%", left: "40%", size: "3rem" },
        { emoji: "🌺", top: "40%", left: "20%", size: "2.5rem" },
    ];

    const containerVariants = {
        initial: { opacity: 0 },
        animate: {
            opacity: 1,
            transition: { duration: 0.5 }
        },
        exit: {
            opacity: 0,
            transition: { duration: 0.5 }
        }
    };

    const contentVariants = {
        initial: { opacity: 0, y: 20 },
        animate: {
            opacity: 1,
            y: 0,
            transition: {
                duration: 0.8,
                staggerChildren: 0.2
            }
        }
    };

    const itemVariants = {
        initial: { opacity: 0, y: 20 },
        animate: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.5 }
        }
    };

    const floatingAnimation = {
        animate: {
            y: [0, -20, 0],
            transition: {
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                ease: "easeInOut"
            }
        }
    };

    return (
        <WelcomeContainer
            variants={containerVariants}
            initial="initial"
            animate="animate"
            exit="exit"
        >
            <BackgroundElements>
                {floatingElements.map((element, index) => (
                    <FloatingElement
                        key={index}
                        style={{
                            top: element.top,
                            bottom: element.bottom,
                            left: element.left,
                            right: element.right,
                            fontSize: element.size
                        }}
                        animate={floatingAnimation.animate}
                    >
                        {element.emoji}
                    </FloatingElement>
                ))}
            </BackgroundElements>

            <ContentWrapper>
                <motion.div
                    variants={contentVariants}
                    initial="initial"
                    animate="animate"
                >
                    <WelcomeTitle variants={itemVariants}>
                        Welcome to Ceylon Guide 🇱🇰
                    </WelcomeTitle>

                    <WelcomeSubtitle variants={itemVariants}>
                        Your AI-powered companion for exploring the wonders of Sri Lanka
                    </WelcomeSubtitle>

                    <FeaturesGrid>
                        {features.map((feature, index) => (
                            <FeatureCard
                                key={index}
                                variants={itemVariants}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <FeatureIcon>{feature.icon}</FeatureIcon>
                                <FeatureTitle>{feature.title}</FeatureTitle>
                                <FeatureDescription>{feature.description}</FeatureDescription>
                            </FeatureCard>
                        ))}
                    </FeaturesGrid>

                    <GetStartedButton
                        onClick={onComplete}
                        variants={itemVariants}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        Start Exploring
                        <FiArrowRight />
                    </GetStartedButton>
                </motion.div>

                <ProgressIndicator>
                    {[0, 1, 2].map((step) => (
                        <ProgressDot
                            key={step}
                            active={currentStep >= step}
                            onClick={() => setCurrentStep(step)}
                            whileHover={{ scale: 1.2 }}
                        />
                    ))}
                </ProgressIndicator>
            </ContentWrapper>
        </WelcomeContainer>
    );
};

export default WelcomeScreen;
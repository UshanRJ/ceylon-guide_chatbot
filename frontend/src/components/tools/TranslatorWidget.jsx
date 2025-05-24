// frontend/src/components/tools/TranslatorWidget.jsx
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiGlobe, FiRefreshCw, FiCopy, FiCheck } from 'react-icons/fi';
import chatService from '../../services/chatService';

const Container = styled.div`
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  color: ${props => props.theme.colors.gray[900]};
  margin-bottom: 1.5rem;
`;

const TranslatorCard = styled.div`
  background: ${props => props.theme.colors.gray[50]};
  border: 1px solid ${props => props.theme.colors.gray[200]};
  border-radius: 0.75rem;
  padding: 1.5rem;
`;

const LanguageSelector = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const LanguageSelect = styled.select`
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  font-size: 1rem;
  background: white;
  cursor: pointer;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const SwapButton = styled.button`
  background: ${props => props.theme.colors.gray[100]};
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  padding: 0.5rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.gray[200]};
  }
`;

const TextArea = styled.textarea`
  width: 100%;
  padding: 1rem;
  border: 1px solid ${props => props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  font-size: 1rem;
  resize: vertical;
  min-height: 120px;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const TranslateButton = styled.button`
  width: 100%;
  padding: 0.75rem 1.5rem;
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 1rem 0;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.secondary};
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ResultCard = styled.div`
  background: white;
  border: 2px solid ${props => props.theme.colors.primary};
  border-radius: 0.5rem;
  padding: 1rem;
  margin-top: 1rem;
  position: relative;
`;

const ResultText = styled.div`
  font-size: 1.125rem;
  color: ${props => props.theme.colors.gray[900]};
  line-height: 1.6;
  padding-right: 2.5rem;
`;

const CopyButton = styled.button`
  position: absolute;
  top: 1rem;
  right: 1rem;
    background-color: ${props => props.$copied ? '#ccc' : '#eee'};
  color: ${props => props.copied ? 'white' : props.theme.colors.gray[700]};
  border: 1px solid ${props => props.copied ? props.theme.colors.success : props.theme.colors.gray[300]};
  border-radius: 0.5rem;
  padding: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.copied ? props.theme.colors.success : props.theme.colors.gray[200]};
  }
`;

const CommonPhrases = styled.div`
  margin-top: 2rem;
`;

const PhrasesTitle = styled.h3`
  font-size: 1.125rem;
  color: ${props => props.theme.colors.gray[900]};
  margin-bottom: 1rem;
`;

const PhrasesList = styled.div`
  display: grid;
  gap: 0.5rem;
`;

const PhraseItem = styled.button`
  background: white;
  border: 1px solid ${props => props.theme.colors.gray[200]};
  border-radius: 0.5rem;
  padding: 0.75rem;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: ${props => props.theme.colors.gray[50]};
    border-color: ${props => props.theme.colors.primary};
  }
  
  .english {
    font-weight: 500;
    color: ${props => props.theme.colors.gray[900]};
  }
  
  .translation {
    font-size: 0.875rem;
    color: ${props => props.theme.colors.gray[600]};
    margin-top: 0.25rem;
  }
`;

const TranslatorWidget = () => {
    const [sourceText, setSourceText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [sourceLang, setSourceLang] = useState('english');
    const [targetLang, setTargetLang] = useState('sinhala');
    const [loading, setLoading] = useState(false);
    const [copied, setCopied] = useState(false);

    const languages = [
        { code: 'english', name: 'English' },
        { code: 'sinhala', name: 'සිංහල (Sinhala)' },
        { code: 'tamil', name: 'தமிழ் (Tamil)' }
    ];

    const commonPhrases = [
        { en: 'Hello', si: 'ආයුබෝවන් (Ayubowan)', ta: 'வணக்கம் (Vanakkam)' },
        { en: 'Thank you', si: 'ස්තූතියි (Sthuthiyi)', ta: 'நன்றி (Nandri)' },
        { en: 'How are you?', si: 'කොහොමද? (Kohomada?)', ta: 'எப்படி இருக்கிறீர்கள்? (Eppadi irukkirirgal?)' },
        { en: 'Where is...?', si: 'කොහෙද...? (Koheda...?)', ta: 'எங்கே...? (Enge...?)' },
        { en: 'How much?', si: 'කීයද? (Kiyada?)', ta: 'எவ்வளவு? (Evvalavu?)' }
    ];

    const handleTranslate = async () => {
        if (!sourceText.trim()) return;

        setLoading(true);
        try {
            const response = await chatService.translateText(
                sourceText,
                sourceLang,
                targetLang
            );
            if (response.success) {
                setTranslatedText(response.translated_text);
            }
        } catch (error) {
            console.error('Translation error:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSwap = () => {
        setSourceLang(targetLang);
        setTargetLang(sourceLang);
        setSourceText(translatedText);
        setTranslatedText(sourceText);
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(translatedText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handlePhraseClick = (phrase) => {
        setSourceText(phrase.en);
        setSourceLang('english');

        if (targetLang === 'sinhala') {
            setTranslatedText(phrase.si);
        } else if (targetLang === 'tamil') {
            setTranslatedText(phrase.ta);
        }
    };

    return (
        <Container>
            <Title>Translator</Title>

            <TranslatorCard>
                <LanguageSelector>
                    <LanguageSelect
                        value={sourceLang}
                        onChange={(e) => setSourceLang(e.target.value)}
                    >
                        {languages.map((lang) => (
                            <option key={lang.code} value={lang.code}>{lang.name}</option>
                        ))}
                    </LanguageSelect>

                    <SwapButton onClick={handleSwap}>
                        <FiRefreshCw />
                    </SwapButton>

                    <LanguageSelect
                        value={targetLang}
                        onChange={(e) => setTargetLang(e.target.value)}
                    >
                        {languages.map((lang) => (
                            <option key={lang.code} value={lang.code}>{lang.name}</option>
                        ))}
                    </LanguageSelect>
                </LanguageSelector>

                <TextArea
                    value={sourceText}
                    onChange={(e) => setSourceText(e.target.value)}
                    placeholder="Enter text to translate..."
                />

                <TranslateButton onClick={handleTranslate} disabled={loading || !sourceText.trim()}>
                    <FiGlobe />
                    {loading ? 'Translating...' : 'Translate'}
                </TranslateButton>

                {translatedText && (
                    <ResultCard>
                        <ResultText>{translatedText}</ResultText>
                        <CopyButton onClick={handleCopy} copied={copied}>
                            {copied ? <FiCheck /> : <FiCopy />}
                        </CopyButton>
                    </ResultCard>
                )}
            </TranslatorCard>

            <CommonPhrases>
                <PhrasesTitle>Common Phrases</PhrasesTitle>
                <PhrasesList>
                    {commonPhrases.map((phrase, index) => (
                        <PhraseItem key={index} onClick={() => handlePhraseClick(phrase)}>
                            <div className="english">{phrase.en}</div>
                            <div className="translation">
                                {targetLang === 'sinhala' ? phrase.si : phrase.ta}
                            </div>
                        </PhraseItem>
                    ))}
                </PhrasesList>
            </CommonPhrases>
        </Container>
    );
};

export default TranslatorWidget;
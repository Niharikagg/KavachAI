# 🛡️ KavachAI

### Contextual Privacy & Integrity Gateway for LLM Pipelines

> **Team:** Phoneix
> **Track:** AI/ML
> **Domain:** AI • NLP • Privacy • Cybersecurity

---

## 📌 1. Project Overview

**KavachAI** is an AI/ML-powered privacy middleware that protects sensitive information before it reaches an LLM or downstream application.

Unlike conventional PII-redaction systems that primarily detect explicit identifiers, KavachAI also considers **contextual and quasi-identifying information** that can increase the risk of re-identification when multiple attributes are combined.

### Core Principle

> **Detect → Assess → Generalize → Protect**

---

## 🚨 2. Problem Statement

LLM-powered applications process large amounts of user-generated information that may contain sensitive data.

Traditional privacy systems generally focus on direct identifiers such as:

* Names
* Phone numbers
* Email addresses
* Aadhaar numbers
* PAN numbers
* Bank account numbers
* Addresses

However, sensitive information is not always a single identifier.

For example:

> *"I live in a small village in Karnataka, have a rare neurological disorder, and I am the only person in my family earning an income."*

Individually, these details may appear harmless. **When combined, they can significantly increase re-identification risk.**

### The Gap

```text
Traditional Privacy
       ↓
Detect direct PII
       ↓
Redact PII
```

**KavachAI**

```text
Direct PII
     +
Contextual Attributes
     +
Specificity
     +
Uniqueness
     ↓
Contextual Privacy Risk
     ↓
Selective Protection
```

---

# 💡 3. Our Solution

KavachAI functions as a **privacy gateway** between the user/application and an LLM pipeline.

```text
┌──────────────────┐
│   User Message   │
└────────┬─────────┘
         ↓
┌─────────────────────────┐
│ Module 1                │
│ Contextual Extraction   │
│                         │
│ • Direct PII            │
│ • Contextual Attributes │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ Module 2                │
│ Privacy Risk Engine     │
│                         │
│ • Feature Extraction    │
│ • Risk Scoring          │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ Module 3                │
│ Privacy Optimizer       │
│                         │
│ • Selective             │
│   Generalization        │
└────────┬────────────────┘
         ↓
┌──────────────────┐
│ Protected Data   │
└────────┬─────────┘
         ↓
   ┌───────────┐
   │ LLM / DB  │
   └───────────┘
```

---

# 🧠 4. System Modules

## Module 1 — Contextual Data Extraction

Identifies both **direct PII** and **contextual information**.

### Direct PII

* 👤 Person
* 📱 Phone number
* 📧 Email
* 🪪 Aadhaar
* 💳 PAN
* 🏦 Bank account
* 🏦 IFSC
* 🚗 Vehicle registration
* 🗳️ Voter ID
* 📍 Address

### Contextual / Quasi-Identifiers

* Age
* Health conditions
* Occupation
* Education
* Location
* Facilities
* Uniqueness indicators
* Other attributes contributing to re-identification

The output is normalized into a structured format containing the **attribute type, value, confidence, specificity, and source message**.

---

## Module 2 — Privacy Risk Engine

The extracted information is converted into privacy-risk features.

KavachAI evaluates factors including:

* **Attribute count**
* **Average specificity**
* **Maximum specificity**
* **High-specificity attributes**
* **Sensitive categories**
* **Uniqueness / rarity**

The system produces a contextual risk assessment:

```text
🟢 LOW
🟡 MEDIUM
🔴 HIGH
```

This allows the privacy layer to determine the level of transformation required.

---

## Module 3 — Privacy Optimizer

KavachAI does not simply delete every sensitive piece of information.

Instead, it performs **selective generalization**.

### Example

| Original                             | Protected                 |
| ------------------------------------ | ------------------------- |
| `37 years old`                       | `adult in their thirties` |
| `Rare autoimmune disorder`           | `autoimmune disorder`     |
| `18, Jayanagar 4th Block, Bengaluru` | `Bengaluru`               |
| `9988776655`                         | `[PHONE]`                 |
| `kavya@example.com`                  | `[EMAIL]`                 |

The goal is:

> **Reduce privacy risk while preserving useful information.**

After transformation, KavachAI performs a contextual risk reassessment and reports the resulting privacy metrics.

---

# 🔐 5. Key Innovation

### Traditional PII Redaction

```text
Phone Number  → [PHONE]
Email         → [EMAIL]
Aadhaar       → [AADHAAR]
```

### KavachAI

```text
             User Data
                 │
        ┌────────┴────────┐
        ↓                 ↓
    Direct PII      Contextual Data
        │                 │
        └────────┬────────┘
                 ↓
       Re-identification Risk
                 ↓
        Selective Generalization
                 ↓
          Protected Data
```

KavachAI therefore treats privacy as a **contextual risk-management problem**, rather than simple keyword or identifier removal.

---

# 🌐 6. Multilingual Support

KavachAI currently supports:

🇬🇧 **English**
🇮🇳 **Hindi**
🇮🇳 **Kannada**

The NLP pipeline is designed to process sensitive information across these languages while maintaining a common structured output format.

---

# 🤖 7. Track Selection

## **AI/ML**

KavachAI was selected under the **AI/ML track** because its core pipeline uses AI/ML and NLP techniques for:

* Sensitive attribute extraction
* Named entity recognition
* Multilingual NLP
* Contextual attribute identification
* Privacy-risk feature extraction
* Risk prediction
* Context-aware privacy optimization

---

# 🛠️ 8. Technology Stack

| Category                 | Technology                         |
| ------------------------ | ---------------------------------- |
| **Programming Language** | Python                             |
| **NLP / Transformers**   | Hugging Face Transformers          |
| **Multilingual Model**   | MuRIL                              |
| **PII Detection**        | Microsoft Presidio                 |
| **Pattern Detection**    | Custom Regex & Pattern Recognizers |
| **Machine Learning**     | Scikit-learn                       |
| **Risk Model**           | Random Forest                      |
| **Backend**              | FastAPI                            |
| **Database**             | PostgreSQL                         |
| **Frontend**             | HTML, CSS, JavaScript              |
| **Data Format**          | JSON                               |
| **Version Control**      | Git & GitHub                       |

---

# 📊 9. Example

### Input

```text
My name is Kavya Nair, I am 37 years old and live in
Jayanagar, Bengaluru. I have a rare autoimmune disorder
and work as a research scientist at a private biotech company.
```

### Module 1 — Extraction

```text
PERSON
AGE
LOCATION
HEALTH
OCCUPATION
ORGANIZATION
UNIQUENESS
```

### Module 2 — Risk Assessment

```text
       Contextual Analysis
               ↓
           MEDIUM RISK
```

### Module 3 — Protection

```text
My name is [PERSON], I am an adult in their thirties
and live in Bengaluru. I have an autoimmune disorder
and work as a research scientist at a private biotech company.
```

### Result

```text
Specificity ↓
Re-identification Risk ↓
Useful Context ✓
```

---

# 🏗️ 10. Architecture

```text
                    ┌──────────────┐
                    │     USER     │
                    └──────┬───────┘
                           │
                           ▼
              ┌───────────────────────┐
              │       KAVACHAI        │
              │   PRIVACY GATEWAY     │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │       MODULE 1        │
              │ Contextual Extraction │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │       MODULE 2        │
              │    Risk Assessment    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │       MODULE 3        │
              │ Privacy Optimization  │
              └───────────┬───────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ Protected Data │
                 └───────┬────────┘
                         │
                  ┌──────┴──────┐
                  ▼             ▼
                LLM           Database
```

---

# 📈 11. What KavachAI Provides

* 🔎 **Direct PII Detection**
* 🧠 **Contextual Attribute Extraction**
* 📊 **Privacy Risk Scoring**
* 🛡️ **Selective Generalization**
* 🌐 **English, Hindi & Kannada Support**
* 📉 **Risk Reduction Measurement**
* 📋 **Transformation Tracking**
* 🔄 **Model-Agnostic Middleware Architecture**
* ⚙️ **Utility-Preserving Privacy Protection**

---

# 👥 12. Team

## **Phoneix**

### Project

**KavachAI — Contextual Privacy & Integrity Gateway for LLM Pipelines**

### Track

**AI/ML**

### Domain

**AI • NLP • Privacy • Cybersecurity**

---

# 🚀 13. Vision

KavachAI aims to move privacy protection beyond simple PII redaction.

Instead of asking:

> **“Is this text PII?”**

KavachAI asks:

> **“Could this combination of information increase the risk of identifying or exposing an individual?”**

And then:

> **“What is the minimum transformation required to reduce that risk while preserving utility?”**

**KavachAI — Protect the context, not just the identifier.** 🛡️

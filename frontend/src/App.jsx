import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:5000";

const MODULE_CONFIG = {
  product: {
    icon: "🛡️",
    label: "Product Risk",
    eyebrow: "PRODUCT SECURITY INTELLIGENCE",
    title: (
      <>
        Detect suspicious products
        <span> before you buy.</span>
      </>
    ),
    description:
      "Analyze product metadata and identify suspicious patterns using anomaly detection and intelligent risk indicators.",
    emptyTitle: "Product Risk Analysis",
    emptyDescription:
      "Your AI-powered product security assessment will appear here.",
  },

  review: {
    icon: "⭐",
    label: "Fake Reviews",
    eyebrow: "REVIEW AUTHENTICITY INTELLIGENCE",
    title: (
      <>
        Know which reviews are
        <span> worth trusting.</span>
      </>
    ),
    description:
      "Analyze product reviews using machine learning to identify potentially manipulated or suspicious content.",
    emptyTitle: "Review Authenticity Analysis",
    emptyDescription:
      "Your review classification and authenticity assessment will appear here.",
  },

  phishing: {
    icon: "🔗",
    label: "URL Security",
    eyebrow: "PHISHING & LINK SECURITY",
    title: (
      <>
        Check suspicious links
        <span> before you click.</span>
      </>
    ),
    description:
      "Analyze URL structures and lexical patterns to detect potentially malicious or phishing websites.",
    emptyTitle: "URL Security Analysis",
    emptyDescription:
      "Your phishing detection and URL security assessment will appear here.",
  },
};

function App() {
  // ============================================
  // TOP-LEVEL MODULE
  // ============================================

  const [module, setModule] = useState("product");

  // ============================================
  // PRODUCT ANALYSIS MODE
  // ============================================

  const [mode, setMode] = useState("manual");

  // ============================================
  // MANUAL PRODUCT STATE
  // ============================================

  const [product, setProduct] = useState({
    title: "",
    description: "",
    seller: "",
    price: "",
    average_rating: "",
    rating_number: "",
    features: "",
    categories: "",
  });

  // ============================================
  // AMAZON STATE
  // ============================================

  const [amazonInput, setAmazonInput] = useState("");

  // ============================================
  // FAKE REVIEW STATE
  // ============================================

  const [review, setReview] = useState("");

  // ============================================
  // PHISHING URL STATE
  // ============================================

  const [url, setUrl] = useState("");

  // ============================================
  // COMMON STATE
  // ============================================

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const activeModule = MODULE_CONFIG[module];

  // ============================================
  // MANUAL PRODUCT FORM CHANGE
  // ============================================

  function handleChange(e) {
    const { name, value } = e.target;

    setProduct((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  // ============================================
  // RESET
  // ============================================

  function resetAnalysis() {
    setResult(null);
    setError("");
    setAmazonInput("");
    setReview("");
    setUrl("");

    setProduct({
      title: "",
      description: "",
      seller: "",
      price: "",
      average_rating: "",
      rating_number: "",
      features: "",
      categories: "",
    });
  }

  // ============================================
  // MODULE SWITCH
  // ============================================

  function switchModule(newModule) {
    setModule(newModule);

    setResult(null);
    setError("");
    setLoading(false);

    if (newModule === "product") {
      setMode("manual");
    }
  }

  // ============================================
  // MANUAL PRODUCT ANALYSIS
  // ============================================

  async function analyzeManualProduct() {
    if (!product.title.trim()) {
      setError("Product title is required.");
      return;
    }

    if (!product.description.trim()) {
      setError("Product description is required.");
      return;
    }

    if (
      product.average_rating !== "" &&
      (Number(product.average_rating) < 0 ||
        Number(product.average_rating) > 5)
    ) {
      setError("Average rating must be between 0 and 5.");
      return;
    }

    if (
      product.rating_number !== "" &&
      Number(product.rating_number) < 0
    ) {
      setError("Number of ratings cannot be negative.");
      return;
    }

    if (
      product.price !== "" &&
      (Number(product.price) < 0 ||
        Number.isNaN(Number(product.price)))
    ) {
      setError("Please enter a valid non-negative price.");
      return;
    }

    const payload = {
      title: product.title.trim(),

      description: product.description.trim(),

      seller: product.seller.trim(),

      price: product.price || null,

      average_rating:
        product.average_rating !== ""
          ? Number(product.average_rating)
          : null,

      rating_number:
        product.rating_number !== ""
          ? Number(product.rating_number)
          : null,

      features: product.features
        ? product.features
            .split("\n")
            .map((item) => item.trim())
            .filter(Boolean)
        : [],

      categories: product.categories
        ? product.categories
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean)
        : [],

      images: {},
      videos: {},
    };

    const response = await fetch(`${API_URL}/api/products`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to analyze product");
    }

    setResult(data);
  }

  // ============================================
  // AMAZON PRODUCT ANALYSIS
  // ============================================

  async function analyzeAmazonProduct() {
    if (!amazonInput.trim()) {
      setError("Please enter an Amazon product URL or ASIN.");
      return;
    }

    const response = await fetch(`${API_URL}/api/products/url`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url: amazonInput.trim(),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || "Failed to analyze Amazon product"
      );
    }

    setResult({
      ...data.risk,
      parent_asin: data.parent_asin,
      product: data.product,
    });
  }

  // ============================================
  // FAKE REVIEW ANALYSIS
  // ============================================

  async function analyzeFakeReview() {
    if (!review.trim()) {
      setError("Please enter a review to analyze.");
      return;
    }

    const response = await fetch(`${API_URL}/api/reviews`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        review: review.trim(),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || "Failed to analyze review"
      );
    }

    setResult(data);
  }

  // ============================================
  // PHISHING URL ANALYSIS
  // ============================================

  async function analyzePhishingUrl() {
    if (!url.trim()) {
      setError("Please enter a URL to analyze.");
      return;
    }

    const response = await fetch(`${API_URL}/api/urls`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url: url.trim(),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || "Failed to analyze URL"
      );
    }

    setResult(data);
  }

  // ============================================
  // MAIN ANALYZE HANDLER
  // ============================================

  async function analyze(e) {
    e.preventDefault();

    setError("");
    setResult(null);
    setLoading(true);

    try {
      if (module === "product") {
        if (mode === "amazon") {
          await analyzeAmazonProduct();
        } else {
          await analyzeManualProduct();
        }
      }

      if (module === "review") {
        await analyzeFakeReview();
      }

      if (module === "phishing") {
        await analyzePhishingUrl();
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  // ============================================
  // RISK CLASS
  // ============================================

  function getRiskClass(level) {
    return level?.toLowerCase() || "medium";
  }

  return (
    <div className="app">

      {/* ============================================
          NAVBAR
      ============================================ */}

      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">T</div>
          <span>TrustGuard</span>
        </div>

        <div className="nav-status">
          <span className="status-dot pulse-dot"></span>
          AI Risk Engine Online
        </div>
      </header>

      <main className="container">

        {/* ============================================
            HERO
        ============================================ */}

        <section className="hero-section">

          <span className="eyebrow">
            {activeModule.eyebrow}
          </span>

          <h1>
            {activeModule.title}
          </h1>

          <p>
            {activeModule.description}
          </p>

          <div className="security-stats">

            <div>
              <strong>3</strong>
              <span>AI Security Modules</span>
            </div>

            <div>
              <strong>24/7</strong>
              <span>Risk Intelligence</span>
            </div>

            <div>
              <strong>ML</strong>
              <span>Powered Analysis</span>
            </div>

          </div>

        </section>

        {/* ============================================
            TOP-LEVEL MODULE SWITCH
        ============================================ */}

        <div className="module-switch">

          {Object.entries(MODULE_CONFIG).map(([key, config]) => (

            <button
              key={key}
              type="button"
              className={
                module === key
                  ? "module-button active"
                  : "module-button"
              }
              onClick={() => switchModule(key)}
            >

              <span className="module-icon">
                {config.icon}
              </span>

              <span className="module-content">

                <strong>
                  {config.label}
                </strong>

                <small>
                  {key === "product"
                    ? "Detect suspicious products"
                    : key === "review"
                    ? "Identify manipulated reviews"
                    : "Detect phishing websites"}
                </small>

              </span>

              <span className="module-arrow">
                →
              </span>

            </button>

          ))}

        </div>

        <section className="workspace">

          {/* ============================================
              INPUT CARD
          ============================================ */}

          <div className="card input-card">

            <div className="card-header">

              <div>

                <h2>
                  {module === "product"
                    ? "Product Analysis"
                    : module === "review"
                    ? "Fake Review Detection"
                    : "Phishing URL Detection"}
                </h2>

                <p>
                  {module === "product"
                    ? "Choose how you want to analyze the product."
                    : module === "review"
                    ? "Analyze a product review using the trained machine learning model."
                    : "Analyze a URL for potential phishing activity."}
                </p>

              </div>

            </div>

            {/* ============================================
                PRODUCT MODE SWITCH
            ============================================ */}

            {module === "product" && (
              <div className="analysis-mode">

                <button
                  type="button"
                  className={
                    mode === "manual"
                      ? "mode-button active"
                      : "mode-button"
                  }
                  onClick={() => {
                    setMode("manual");
                    setError("");
                    setResult(null);
                  }}
                >
                  Manual Entry
                </button>

                <button
                  type="button"
                  className={
                    mode === "amazon"
                      ? "mode-button active"
                      : "mode-button"
                  }
                  onClick={() => {
                    setMode("amazon");
                    setError("");
                    setResult(null);
                  }}
                >
                  Amazon Product
                </button>

              </div>
            )}

            <form onSubmit={analyze}>

              {/* ============================================
                  AMAZON MODE
              ============================================ */}

              {module === "product" &&
                mode === "amazon" && (
                  <>

                    <div className="amazon-info">

                      <strong>
                        Amazon Product Analysis
                      </strong>

                      <p>
                        Paste an Amazon product URL or enter its
                        10-character ASIN. TrustGuard will
                        retrieve the product metadata from the
                        local Amazon dataset.
                      </p>

                    </div>

                    <div className="form-group">

                      <label>
                        Amazon URL or ASIN
                      </label>

                      <input
                        type="text"
                        value={amazonInput}
                        onChange={(e) =>
                          setAmazonInput(e.target.value)
                        }
                        placeholder="https://www.amazon.com/dp/B07BJ7ZZL7"
                      />

                      <small>
                        Example ASIN: B07BJ7ZZL7
                      </small>

                    </div>

                  </>
                )}

              {/* ============================================
                  MANUAL PRODUCT MODE
              ============================================ */}

              {module === "product" &&
                mode === "manual" && (
                  <>

                    <div className="form-group">

                      <label>
                        Product title
                      </label>

                      <input
                        type="text"
                        name="title"
                        value={product.title}
                        onChange={handleChange}
                        placeholder="e.g. Wireless Bluetooth Headphones"
                      />

                    </div>

                    <div className="form-group">

                      <label>
                        Description
                      </label>

                      <textarea
                        name="description"
                        value={product.description}
                        onChange={handleChange}
                        placeholder="Enter the product description..."
                        rows="4"
                      />

                    </div>

                    <div className="form-row">

                      <div className="form-group">

                        <label>
                          Seller
                        </label>

                        <input
                          type="text"
                          name="seller"
                          value={product.seller}
                          onChange={handleChange}
                          placeholder="Seller name"
                        />

                      </div>

                      <div className="form-group">

                        <label>
                          Price
                        </label>

                        <input
                          type="text"
                          name="price"
                          value={product.price}
                          onChange={handleChange}
                          placeholder="49.99"
                        />

                      </div>

                    </div>

                    <div className="form-row">

                      <div className="form-group">

                        <label>
                          Average rating
                        </label>

                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          max="5"
                          name="average_rating"
                          value={product.average_rating}
                          onChange={handleChange}
                          placeholder="4.2"
                        />

                      </div>

                      <div className="form-group">

                        <label>
                          Number of ratings
                        </label>

                        <input
                          type="number"
                          min="0"
                          name="rating_number"
                          value={product.rating_number}
                          onChange={handleChange}
                          placeholder="150"
                        />

                      </div>

                    </div>

                    <div className="form-group">

                      <label>
                        Features
                      </label>

                      <textarea
                        name="features"
                        value={product.features}
                        onChange={handleChange}
                        placeholder={
                          "Bluetooth connectivity\nNoise cancellation\nLong battery life"
                        }
                        rows="3"
                      />

                      <small>
                        Enter one feature per line.
                      </small>

                    </div>

                    <div className="form-group">

                      <label>
                        Categories
                      </label>

                      <input
                        type="text"
                        name="categories"
                        value={product.categories}
                        onChange={handleChange}
                        placeholder="Electronics, Headphones"
                      />

                      <small>
                        Separate categories with commas.
                      </small>

                    </div>

                  </>
                )}

              {/* ============================================
                  FAKE REVIEW MODE
              ============================================ */}

              {module === "review" && (
                <>

                  <div className="amazon-info">

                    <strong>
                      Fake Review Detection
                    </strong>

                    <p>
                      Enter a product review and TrustGuard
                      will classify it using the trained
                      Linear SVM model.
                    </p>

                  </div>

                  <div className="form-group">

                    <label>
                      Product review
                    </label>

                    <textarea
                      value={review}
                      onChange={(e) =>
                        setReview(e.target.value)
                      }
                      placeholder="Enter the product review..."
                      rows="8"
                    />

                    <small>
                      Enter the complete review you want
                      TrustGuard to analyze.
                    </small>

                  </div>

                </>
              )}

              {/* ============================================
                  PHISHING URL MODE
              ============================================ */}

              {module === "phishing" && (
                <>

                  <div className="amazon-info">

                    <strong>
                      Phishing URL Detection
                    </strong>

                    <p>
                      Enter a website URL and TrustGuard will
                      analyze its structural characteristics
                      using the trained phishing detection model.
                    </p>

                  </div>

                  <div className="form-group">

                    <label>
                      Website URL
                    </label>

                    <input
                      type="text"
                      value={url}
                      onChange={(e) =>
                        setUrl(e.target.value)
                      }
                      placeholder="https://example.com"
                    />

                    <small>
                      Enter the complete URL including https:// when available.
                    </small>

                  </div>

                </>
              )}

              {/* ============================================
                  ERROR
              ============================================ */}

              {error && (
                <div className="error-box">
                  {error}
                </div>
              )}

              {/* ============================================
                  ANALYZE BUTTON
              ============================================ */}

              <button
                type="submit"
                className="analyze-button"
                disabled={loading}
              >
                {loading
                  ? "Analyzing..."
                  : module === "product"
                  ? mode === "amazon"
                    ? "Analyze Amazon Product"
                    : "Analyze Product"
                  : module === "review"
                  ? "Analyze Review"
                  : "Analyze URL"}
              </button>

            </form>

          </div>

          {/* ============================================
              RESULT CARD
          ============================================ */}

          <div className="card result-card">

            {!result && !loading && (
              <div className="empty-result">

                <div className="empty-icon">
                  {activeModule.icon}
                </div>

                <div className="empty-status">
                  READY FOR ANALYSIS
                </div>

                <h2>
                  {activeModule.emptyTitle}
                </h2>

                <p>
                  {activeModule.emptyDescription}
                </p>

                <div className="analysis-flow">

                  <div>
                    <span>01</span>
                    <small>Submit</small>
                  </div>

                  <i>→</i>

                  <div>
                    <span>02</span>
                    <small>Analyze</small>
                  </div>

                  <i>→</i>

                  <div>
                    <span>03</span>
                    <small>Assess</small>
                  </div>

                </div>

              </div>
            )}

            {loading && (
              <div className="empty-result">

                <div className="loader"></div>

                <h2>
                  Analyzing...
                </h2>

                <p>
                  TrustGuard is processing the input
                  using the machine learning model.
                </p>

              </div>
            )}

            {/* ============================================
                FAKE REVIEW RESULT
            ============================================ */}

            {result &&
              !loading &&
              module === "review" && (
                <div className="result-content">

                  <div className="result-header">

                    <div>

                      <span className="eyebrow">
                        ANALYSIS COMPLETE
                      </span>

                      <h2>
                        Review Classification
                      </h2>

                    </div>

                    <span
                      className={`risk-badge ${
                        result.verdict === "Fake Review"
                          ? "high"
                          : "low"
                      }`}
                    >
                      {result.verdict}
                    </span>

                  </div>

                  <div className="score-section">

                    <div
                      className={`score-circle ${
                        result.verdict === "Fake Review" ? "high" : "low"
                      }`}
                      style={{
                        "--score":
                          result.verdict === "Fake Review"
                            ? "75%"
                            : "25%",
                      }}
                    >

                      <div>

                        <strong>
                          {result.verdict === "Fake Review"
                            ? "!"
                            : "✓"}
                        </strong>

                      </div>

                    </div>

                    <div className="score-info">

                      <span>
                        CLASSIFICATION
                      </span>

                      <h3>
                        {result.verdict}
                      </h3>

                      <p className="risk-summary">
                        {result.verdict === "Fake Review"
                          ? "The review has been classified as potentially fake by the trained Linear SVM model."
                          : "The review has been classified as genuine by the trained Linear SVM model."}
                      </p>

                    </div>

                  </div>

                  <div className="section-divider"></div>

                  <section className="explanations">

                    <h3>
                      Analysis Result
                    </h3>

                    <div className="explanation">

                      <div className="explanation-icon">
                        !
                      </div>

                      <div>

                        <strong>
                          Model Prediction
                        </strong>

                        <p>
                          {result.reason}
                        </p>

                      </div>

                    </div>

                  </section>

                  <section className="model-details">

                    <h3>
                      Model Details
                    </h3>

                    <div className="model-grid">

                      <div>

                        <span>
                          Model
                        </span>

                        <strong>
                          Linear SVM
                        </strong>

                      </div>

                      <div>

                        <span>
                          Feature Extraction
                        </span>

                        <strong>
                          TF-IDF
                        </strong>

                      </div>

                      <div>

                        <span>
                          Prediction
                        </span>

                        <strong>
                          {result.verdict}
                        </strong>

                      </div>

                    </div>

                  </section>

                  <button
                    type="button"
                    className="reset-button"
                    onClick={resetAnalysis}
                  >
                    Analyze Another Review
                  </button>

                </div>
              )}

            {/* ============================================
                PHISHING URL RESULT
            ============================================ */}

            {result &&
              !loading &&
              module === "phishing" && (
                <div className="result-content">

                  <div className="result-header">

                    <div>

                      <span className="eyebrow">
                        ANALYSIS COMPLETE
                      </span>

                      <h2>
                        URL Classification
                      </h2>

                    </div>

                    <span
                      className={`risk-badge ${
                        result.verdict?.toLowerCase().includes("phishing")
                          ? "high"
                          : "low"
                      }`}
                    >
                      {result.verdict}
                    </span>

                  </div>

                  <div className="score-section">

                    <div
                      className={`score-circle ${
                        result.verdict?.toLowerCase().includes("phishing")
                          ? "high"
                          : "low"
                      }`}
                      style={{
                        "--score":
                          result.verdict?.toLowerCase().includes("phishing")
                            ? `${Math.min(100, Number(result.confidence) || 75)}%`
                            : `${Math.max(
                                10,
                                100 - (Number(result.confidence) || 50)
                              )}%`,
                      }}
                    >

                      <div>

                        <strong>
                          {result.verdict
                            ?.toLowerCase()
                            .includes("phishing")
                            ? "!"
                            : "✓"}
                        </strong>

                      </div>

                    </div>

                    <div className="score-info">

                      <span>
                        CLASSIFICATION
                      </span>

                      <h3>
                        {result.verdict}
                      </h3>

                      <p className="risk-summary">

                        {result.verdict
                          ?.toLowerCase()
                          .includes("phishing")
                          ? "The URL has been classified as potentially malicious by TrustGuard's phishing detection model."
                          : "The URL has been classified as legitimate by TrustGuard's phishing detection system."}

                      </p>

                    </div>

                  </div>

                  <div className="metrics">

                    <div className="metric">

                      <span>
                        Confidence
                      </span>

                      <strong>
                        {Number(result.confidence).toFixed(2)}%
                      </strong>

                    </div>

                    <div className="metric">

                      <span>
                        Classification
                      </span>

                      <strong>
                        {result.verdict}
                      </strong>

                    </div>

                    <div className="metric">

                      <span>
                        Model
                      </span>

                      <strong>
                        Phishing Classifier
                      </strong>

                    </div>

                  </div>

                  <div className="section-divider"></div>

                  <section className="explanations">

                    <h3>
                      Analysis Result
                    </h3>

                    <div className="explanation">

                      <div className="explanation-icon">
                        !
                      </div>

                      <div>

                        <strong>
                          Model Prediction
                        </strong>

                        <p>
                          {result.reason}
                        </p>

                      </div>

                    </div>

                  </section>

                  <section className="model-details">

                    <h3>
                      Model Details
                    </h3>

                    <div className="model-grid">

                      <div>

                        <span>
                          Model
                        </span>

                        <strong>
                          Phishing Classifier
                        </strong>

                      </div>

                      <div>

                        <span>
                          Prediction
                        </span>

                        <strong>
                          {result.verdict}
                        </strong>

                      </div>

                      <div>

                        <span>
                          Confidence
                        </span>

                        <strong>
                          {Number(result.confidence).toFixed(2)}%
                        </strong>

                      </div>

                    </div>

                  </section>

                  <button
                    type="button"
                    className="reset-button"
                    onClick={resetAnalysis}
                  >
                    Analyze Another URL
                  </button>

                </div>
              )}

            {/* ============================================
                PRODUCT RESULT
            ============================================ */}

            {result &&
              !loading &&
              module === "product" && (
                <div className="result-content">

                  <div className="result-header">

                    <div>

                      <span className="eyebrow">
                        ANALYSIS COMPLETE
                      </span>

                      <h2>
                        Product Risk Assessment
                      </h2>

                    </div>

                    <span
                      className={`risk-badge ${getRiskClass(
                        result.risk_level
                      )}`}
                    >
                      {result.risk_level}
                    </span>

                  </div>

                  <div className="score-section">

                    <div
                      className={`score-circle ${getRiskClass(result.risk_level)}`}
                      style={{
                        "--score": `${Math.min(
                          100,
                          Math.max(0, Number(result.risk_score) || 0)
                        )}%`,
                      }}
                    >

                      <div>

                        <strong>
                          {Number(
                            result.risk_score
                          ).toFixed(0)}
                        </strong>

                        <span>
                          /100
                        </span>

                      </div>

                    </div>

                    <div className="score-info">

                      <span>
                        RISK SCORE
                      </span>

                      <h3>
                        {Number(
                          result.risk_score
                        ).toFixed(2)}
                      </h3>

                      <p className="risk-summary">

                        {result.is_anomaly
                          ? "The product shows anomalous metadata patterns according to the model."
                          : result.anomaly_score >= 70
                          ? "The product is not classified as anomalous, but its metadata shows elevated deviation from the reference distribution."
                          : "The product does not show significant metadata anomalies according to the model."}

                      </p>

                    </div>

                  </div>

                  <div className="metrics">

                    <div className="metric">

                      <span>
                        Anomaly Score
                      </span>

                      <strong>
                        {Number(
                          result.anomaly_score
                        ).toFixed(2)}
                      </strong>

                    </div>

                    <div className="metric">

                      <span>
                        Indicator Score
                      </span>

                      <strong>
                        {Number(
                          result.indicator_score
                        ).toFixed(2)}
                      </strong>

                    </div>

                    <div className="metric">

                      <span>
                        Isolation Forest
                      </span>

                      <strong>
                        {result.is_anomaly
                          ? "Anomaly"
                          : "Normal"}
                      </strong>

                    </div>

                  </div>

                  <div className="section-divider"></div>

                  <section className="explanations">

                    <h3>
                      Why this result?
                    </h3>

                    {result.explanations?.length === 0 && (
                      <p className="no-explanations">
                        No specific risk indicators were triggered.
                      </p>

                    )}

                    {result.explanations?.map(
                      (item, index) => (
                        <div
                          className="explanation"
                          key={index}
                        >

                          <div className="explanation-icon">
                            !
                          </div>

                          <div>

                            <strong>

                              {item.indicator ===
                              "isolation_forest"
                                ? result.is_anomaly
                                  ? "Metadata anomaly detected"
                                  : "Elevated metadata deviation"
                                : item.indicator ===
                                  "missing_seller"
                                ? "Missing seller information"
                                : item.indicator ===
                                  "high_rating_low_reviews"
                                ? "High rating with very few reviews"
                                : item.indicator ===
                                  "extreme_rating"
                                ? "Extreme product rating"
                                : item.indicator ===
                                  "very_low_review_count"
                                ? "Very low review count"
                                : item.indicator ===
                                  "very_short_title"
                                ? "Very short product title"
                                : item.indicator ===
                                  "very_short_description"
                                ? "Very short description"
                                : item.indicator}

                            </strong>

                            <p>

                              {item.indicator ===
                              "isolation_forest"
                                ? result.is_anomaly
                                  ? "Product metadata is highly anomalous relative to the reference metadata distribution."
                                  : "Product metadata is relatively unusual compared with the reference distribution, but the model did not classify it as an anomaly."
                                : item.explanation}

                            </p>

                          </div>

                        </div>
                      )
                    )}

                  </section>

                  <section className="model-details">

                    <h3>
                      Model Details
                    </h3>

                    <div className="model-grid">

                      <div>

                        <span>
                          Prediction
                        </span>

                        <strong>
                          {
                            result
                              .isolation_forest
                              .prediction
                          }
                        </strong>

                      </div>

                      <div>

                        <span>
                          Decision Score
                        </span>

                        <strong>
                          {
                            result
                              .isolation_forest
                              .decision_score
                          }
                        </strong>

                      </div>

                      <div>

                        <span>
                          Model Weight
                        </span>

                        <strong>
                          {
                            result
                              .risk_weights
                              .isolation_forest
                          }
                        </strong>

                      </div>

                      <div>

                        <span>
                          Risk Indicator Weight
                        </span>

                        <strong>
                          {
                            result
                              .risk_weights
                              .risk_indicators
                          }
                        </strong>

                      </div>

                    </div>

                  </section>

                  <button
                    type="button"
                    className="reset-button"
                    onClick={resetAnalysis}
                  >
                    Analyze Another Product
                  </button>

                </div>
              )}

          </div>

        </section>

        {/* ============================================
            HOW IT WORKS SECTION
        ============================================ */}

        <section className="how-it-works">

          <div className="how-header">

            <span className="eyebrow">
              TRUSTGUARD INTELLIGENCE
            </span>

            <h2>
              How TrustGuard protects your decisions
            </h2>

            <p>
              Multiple AI-powered security modules work together
              to help you identify digital risks before they affect you.
            </p>

          </div>

          <div className="steps-grid">

            <div className="step-card">
              <div className="step-number">01</div>
              <div className="step-icon">⌨️</div>
              <h3>Submit</h3>
              <p>
                Enter a product, review, Amazon ASIN, or website URL.
              </p>
            </div>

            <div className="step-card">
              <div className="step-number">02</div>
              <div className="step-icon">⚙️</div>
              <h3>Analyze</h3>
              <p>
                TrustGuard extracts meaningful signals and patterns
                from your input.
              </p>
            </div>

            <div className="step-card">
              <div className="step-number">03</div>
              <div className="step-icon">🧠</div>
              <h3>AI Detection</h3>
              <p>
                Machine learning models identify anomalies,
                suspicious patterns, and potential risks.
              </p>
            </div>

            <div className="step-card">
              <div className="step-number">04</div>
              <div className="step-icon">🛡️</div>
              <h3>Make Better Decisions</h3>
              <p>
                Understand the detected risk and make safer,
                more informed decisions.
              </p>
            </div>

          </div>

        </section>

      </main>

      <footer>

        <span>
          TrustGuard
        </span>

        <span>
          AI-based product scam detection
        </span>

      </footer>

    </div>
  );
}

export default App;

const axios = require("axios");

const FASTAPI_URL = process.env.FASTAPI_URL;
const PRODUCT_FASTAPI_URL = process.env.PRODUCT_FASTAPI_URL;


// ============================================================
// FAKE REVIEW
// ============================================================

async function predictReview(review) {

    const response = await axios.post(
        `${FASTAPI_URL}/api/v1/predict/review`,
        {
            review: review
        }
    );

    return response.data;
}


// ============================================================
// PHISHING URL
// ============================================================

async function predictUrl(url) {

    const response = await axios.post(
        `${FASTAPI_URL}/api/v1/predict/url`,
        {
            url: url
        }
    );

    return response.data;
}


// ============================================================
// PRODUCT RISK
// ============================================================

async function predictProduct(product) {

    // Convert seller -> store for Python model compatibility
    const payload = {
        ...product,
        store: product.seller || product.store || ""
    };

    delete payload.seller;

    const response = await axios.post(
        `${PRODUCT_FASTAPI_URL}/predict`,
        payload
    );

    return response.data;
}


module.exports = {
    predictReview,
    predictUrl,
    predictProduct
};
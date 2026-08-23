const fs = require("fs");
const path = require("path");

// ============================================================
// AMAZON METADATA JSON LOOKUP
// ============================================================

const LOOKUP_PATH = path.resolve(
    __dirname,
    "../../../product_scam_ml/data/raw/amazon_metadata_lookup.json"
);

let productLookup = null;


// ============================================================
// LOAD AMAZON METADATA
// ============================================================

function loadProductLookup() {

    if (productLookup) {
        return productLookup;
    }

    if (!fs.existsSync(LOOKUP_PATH)) {

        throw new Error(
            `Amazon metadata lookup file not found: ${LOOKUP_PATH}`
        );

    }

    const rawData = fs.readFileSync(
        LOOKUP_PATH,
        "utf-8"
    );

    productLookup = JSON.parse(rawData);

    console.log(
        `Amazon metadata lookup loaded: ${Object.keys(productLookup).length} products`
    );

    return productLookup;
}


// ============================================================
// NORMALIZE TEXT
// ============================================================

function normalizeText(value) {

    if (
        value === null ||
        value === undefined ||
        value === "None"
    ) {
        return "";
    }

    if (Array.isArray(value)) {

        return value
            .filter(
                item =>
                    item !== null &&
                    item !== undefined &&
                    item !== "None"
            )
            .map(item => String(item))
            .join(" ");
    }

    return String(value);
}


// ============================================================
// NORMALIZE ARRAY
// ============================================================

function normalizeArray(value) {

    if (
        value === null ||
        value === undefined ||
        value === "None"
    ) {
        return [];
    }

    if (Array.isArray(value)) {

        return value
            .filter(
                item =>
                    item !== null &&
                    item !== undefined &&
                    item !== "None"
            )
            .map(item => String(item));
    }

    return [String(value)];
}


// ============================================================
// NORMALIZE NULLABLE VALUE
// ============================================================

function normalizeNullable(value) {

    if (
        value === null ||
        value === undefined ||
        value === "None" ||
        value === "NaN"
    ) {
        return null;
    }

    return value;
}


// ============================================================
// EXTRACT AMAZON ASIN
// ============================================================

function extractAmazonAsin(input) {

    if (!input || typeof input !== "string") {
        return null;
    }

    const value = input.trim();

    // --------------------------------------------------------
    // DIRECT ASIN
    // --------------------------------------------------------

    if (/^[A-Z0-9]{10}$/i.test(value)) {
        return value.toUpperCase();
    }

    // --------------------------------------------------------
    // AMAZON PRODUCT URL
    // --------------------------------------------------------

    try {

        const url = new URL(value);

        const match = url.pathname.match(
            /\/(?:dp|gp\/product|gp\/aw\/d)\/([A-Z0-9]{10})(?:[/?]|$)/i
        );

        if (match) {
            return match[1].toUpperCase();
        }

    } catch (error) {

        // Not a valid URL.
        // Since it wasn't a valid direct ASIN either,
        // return null.

    }

    return null;
}


// ============================================================
// FIND PRODUCT BY ASIN
// ============================================================

async function findProductByAsin(asin) {

    if (!asin) {
        return null;
    }

    const lookup = loadProductLookup();

    const record = lookup[asin.toUpperCase()];

    if (!record) {
        return null;
    }

    const product = {
        ...record,

        title: normalizeText(record.title),

        description: normalizeText(record.description),

        features: normalizeArray(record.features),

        categories: normalizeArray(record.categories),

        images: record.images || {},

        videos: record.videos || {},

        seller: normalizeText(record.store),

        price: normalizeNullable(record.price),

        average_rating:
            normalizeNullable(record.average_rating),

        rating_number:
            normalizeNullable(record.rating_number)
    };

    return product;
}


// ============================================================
// EXPORTS
// ============================================================

module.exports = {

    extractAmazonAsin,

    findProductByAsin

};
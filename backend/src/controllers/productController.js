const {
    predictProduct
} = require("../services/mlService");

const {
    extractAmazonAsin,
    findProductByAsin
} = require("../services/productService");


async function analyzeProduct(req, res) {

    try {

        const {
            title,
            description,
            features,
            categories,
            images,
            videos,
            seller,
            price,
            average_rating,
            rating_number
        } = req.body;


        // ====================================================
        // BASIC VALIDATION
        // ====================================================

        if (!title || !title.trim()) {

            return res.status(400).json({
                error: "Product title cannot be empty"
            });

        }


        // ====================================================
        // BUILD PRODUCT OBJECT
        // ====================================================

        const product = {

            title,
            description: description || "",
            features: features || [],
            categories: categories || [],
            images: images || {},
            videos: videos || {},
            store: seller || "",
            price: price ?? null,
            average_rating: average_rating ?? null,
            rating_number: rating_number ?? null

        };


        // ====================================================
        // CALL FASTAPI
        // ====================================================

        const result = await predictProduct(product);

        return res.json(result);

    } catch (error) {

        console.error(
            "Product risk prediction error:",
            error.message
        );

        return res.status(500).json({
            error: "Failed to analyze product"
        });

    }
}

// ============================================================
// ANALYZE AMAZON PRODUCT URL
// ============================================================

async function analyzeProductUrl(req, res) {

    try {

        const { url } = req.body;


        // ====================================================
        // VALIDATE URL
        // ====================================================

        if (!url || !url.trim()) {

            return res.status(400).json({
                error: "Product URL cannot be empty"
            });

        }


        // ====================================================
        // EXTRACT ASIN
        // ====================================================

        const asin = extractAmazonAsin(url.trim());


        if (!asin) {

            return res.status(400).json({
                error:
                    "Invalid Amazon product URL. Could not extract ASIN."
            });

        }


        console.log(
            `Amazon product requested: ${asin}`
        );


        // ====================================================
        // FIND PRODUCT IN LOCAL DATASET
        // ====================================================

        const metadata =
            await findProductByAsin(asin);


        if (!metadata) {

            return res.status(404).json({
                error:
                    `Product ${asin} was not found in the local Amazon metadata dataset.`,
                parent_asin: asin
            });

        }


        // ====================================================
        // BUILD RISK ENGINE INPUT
        // ====================================================

        const product = {

            parent_asin: asin,

            title: metadata.title || "",

            description:
                metadata.description || "",

            features:
                metadata.features || [],

            categories:
                metadata.categories || [],

            images:
                metadata.images || {},

            videos:
                metadata.videos || {},

            store:
                metadata.seller || "",

            price:
                metadata.price ?? null,

            average_rating:
                metadata.average_rating ?? null,

            rating_number:
                metadata.rating_number ?? null
        };


        // ====================================================
        // EXISTING RISK ENGINE
        // ====================================================

        console.log("\n================ AMAZON PRODUCT DEBUG ================");
        console.log("ASIN:", asin);
        console.log("TITLE:", product.title);
        console.log("DESCRIPTION TYPE:", typeof product.description);
        console.log("DESCRIPTION LENGTH:", product.description?.length);
        console.log("FEATURES:", product.features);
        console.log("FEATURES TYPE:", Array.isArray(product.features) ? "array" : typeof product.features);
        console.log("CATEGORIES:", product.categories);
        console.log("IMAGES:", product.images);
        console.log("VIDEOS:", product.videos);
        console.log("STORE:", product.store);
        console.log("PRICE:", product.price);
        console.log("AVERAGE RATING:", product.average_rating);
        console.log("RATING NUMBER:", product.rating_number);
        console.log("========================================================\n");

        const result =
            await predictProduct(product);


        // ====================================================
        // RETURN RESULT
        // ====================================================

        return res.json({

            parent_asin: asin,

            product: {
                title: product.title,
                seller: product.store,
                price: product.price,
                average_rating: product.average_rating,
                rating_number: product.rating_number
            },

            risk: result

        });

    } catch (error) {

        console.error(
            "Amazon product URL analysis error:",
            error
        );

        return res.status(500).json({
            error: "Failed to analyze Amazon product",
            details: error.message
        });

    }
    
}


module.exports = {
    analyzeProduct,
    analyzeProductUrl
};
/**
 * Ergo blockchain helper functions for JavaScript/Node.js
 *
 * This module provides utilities for working with Ergo blockchain data.
 */

const axios = require('axios');

class ErgoClient {
    constructor(nodeUrl = 'http://localhost:9053') {
        this.nodeUrl = nodeUrl.replace(/\/$/, '');
    }

    async getNodeInfo() {
        try {
            const response = await axios.get(`${this.nodeUrl}/info`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get node info: ${error.message}`);
        }
    }

    async getCurrentHeight() {
        const info = await this.getNodeInfo();
        return info.fullHeight || 0;
    }

    async getMempool() {
        try {
            const response = await axios.get(`${this.nodeUrl}/transactions/unconfirmed`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get mempool: ${error.message}`);
        }
    }
}

// Utility functions
const validateErgoAddress = (address) => {
    return typeof address === 'string' &&
           address.length > 40 &&
           /^[9|3|2]/.test(address);
};

const nanoErgToErg = (nanoErg) => {
    return nanoErg / 1_000_000_000;
};

const ergToNanoErg = (erg) => {
    return Math.floor(erg * 1_000_000_000);
};

module.exports = {
    ErgoClient,
    validateErgoAddress,
    nanoErgToErg,
    ergToNanoErg
};

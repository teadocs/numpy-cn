'use strict';
const path = require('path')

module.exports = {
    doc: {
        name: "NumPy 中文文档",
        description: "这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。",
        version: "1.14.0",
        dir: "",
        outDir: "",
        staticDir: ""
    }, 
    theme: {
        dir: "", 
        title: "NumPy 中文文档",
        headHtml: `
        <meta name="description" content="这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。" />
        <meta name="keywords" content="NumPy教程, NumPy文档, NumPy中文文档, NumPy, Python, Python科学计算, 机器学习科学计算, 深度学习科学计算" />
        `,
        footHtml: "",
        isMinify: false, 
        rootPath: "/"
    },
    nav: {
        tree: "./tree"
    }
}
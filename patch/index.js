const fs = require('fs');
const path = require('path');

const modules = 'node_modules';
const vuepress_plugin_comment = 'vuepress-plugin-comment';
const valine = 'valine/dist';
const vuepress = 'vuepress';

const maps = [
  {
    sourceFile: path.resolve(path.join('./file', 'Comment.vue')),
    targetFile: path.resolve(path.join('../', modules, vuepress_plugin_comment, 'Comment.vue')),
    targetDir: path.resolve(path.join('../', modules, vuepress_plugin_comment))
  },
  {
    sourceFile: path.resolve(path.join('./file', 'Valine.min.js')),
    targetFile: path.resolve(path.join('../', modules, valine, 'Valine.min.js')),
    targetDir: path.resolve(path.join('../', modules, valine))
  },
  {
    sourceFile: path.resolve(path.join('./file', 'Valine.Pure.min.js')),
    targetFile: path.resolve(path.join('../', modules, valine, 'Valine.Pure.min.js')),
    targetDir: path.resolve(path.join('../', modules, valine))
  }
];

// 修复留言板的一些bug
for (const item of maps) {
  console.log('targetFile', item.targetFile);
  if (fs.existsSync(item.targetFile)) {
    fs.unlinkSync(item.targetFile);
  }
  const sourceContent = fs.readFileSync(item.sourceFile);
  fs.writeFileSync(item.targetFile, sourceContent);
}

// 替换 cli 文件
const cliPath = path.resolve(path.join('../', modules, vuepress, 'cli.js'));
console.log('cliPath', cliPath);
let sourceContent = fs.readFileSync(cliPath, 'utf-8');
let tempArray = sourceContent.split('\n');
tempArray[0] = `#!/usr/bin/env node --max_old_space_size=4906`;
sourceContent = tempArray.join('\n');
fs.writeFileSync(cliPath, sourceContent);

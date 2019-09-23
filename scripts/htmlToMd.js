// @ts-nocheck
function tempParse(el) {
  let as = $(el).find('a');
  let aStr = '';
  as.each(function (index, el) {
    let $el = $(el);
    let href = $el.attr('href');
    aStr = aStr + `[${$el.text()}](${href})、`;
  });
  console.log(aStr);
}

{
  class Convert {
    constructor(options) {
      this.baseUrl = options.baseUrl;
      this.baseImgDir = options.baseImgDir;
      this.$ = options.$;
      this.mainEl = this.$(options.el);
    }

    getChilden(el) {
      return el.children();
    }

    getMarkdown(el) {
      let that = this;
      let mainEl = el ? this.$(el) : this.mainEl;
      let cd = mainEl.children();
      let content = '';
      cd.each(function (index, el) {
        content += that.parse(el);
      });
      return content;
    }

    parse(el) {
      let nodeName = el.nodeName;
      let content = '';
      switch (nodeName) {
        case 'SPAN':
          content = '';
          break;
        case 'H1':
          content = `# ${this.getHtext(el)}\n`;
          break;
        case 'H2':
          content = `\n## ${this.getHtext(el)}\n`;
          break;
        case 'H3':
          content = `\n### ${this.getHtext(el)}\n`;
          break;
        case 'H4':
          content = `\n#### ${this.getHtext(el)}\n`;
          break;
        case 'H5':
          content = `\n##### ${this.getHtext(el)}\n`;
          break;
        case 'TABLE': {
          let tempContent = this.$(el).prop("innerText");
          tempContent = tempContent.replace(new RegExp(/(	)/g), ' | ');
          let tempArr = tempContent.split('\n');
          let tempArr2 = tempArr[0].split('|');
          let tempArr3 = [];
          Array.from({
            length: tempArr2.length
          }).forEach(() => tempArr3.push('---'));
          tempArr.splice(1, 0, tempArr3.join('|'));
          tempContent = `\n${tempArr.join('\n')}\n`;
          content = this.replaceAForTable(tempContent, this.$(el));
        }
        break;
      case 'P': {
        let className = this.$(el).attr('class');
        if (className !== 'first admonition-title') {
          content = this.getPcontent(this.$(el));
        }
      }
      break;
      case 'IMG': {
        let imgSrc = this.$(el).attr('src');
        let imgNameArr = imgSrc.split('/');
        let imgName = imgNameArr[imgNameArr.length - 1];
        let fileName = imgName.split('.')[0];
        content = `\n![${fileName}](${this.baseImgDir}${imgName})\n`;
      }
      break;
      case 'DIV': {
        let className = this.$(el).attr('class');
        if (!className) {
          content = this.getMarkdown(el);
        } else if (className === 'math notranslate nohighlight') {
          content = `\n<div class="math notranslate nohighlight">\nIt's a mathematical formula.\n</div>\n`;
        } else if (className === 'admonition note') {
          let pEl = this.$(el).find('.first.admonition-title');
          let title = pEl.text();
          let tempContent = this.getMarkdown(el);
          content = `\n::: tip ${title}\n${tempContent}\n:::\n`;
        } else if (className === 'admonition seealso') {
          let tempContent = this.getMarkdown(el);
          content = `\n::: tip See also\n${tempContent}\n:::\n`;
        } else if (className === 'admonition warning') {
          let pEl = this.$(el).find('.first.admonition-title');
          let title = pEl.text();
          let tempContent = this.getMarkdown(el);
          content = `\n::: danger ${title}\n${tempContent}\n:::\n`;
        } else if (className === 'section' || className === 'toctree-wrapper compound') {
          content = this.getMarkdown(el);
        } else if (className.indexOf('highlight-') !== -1) {
          let langName = className.split('highlight-')[1].split(' ')[0];
          if (langName === 'ipython' || langName === 'default' || langName === 'ipython3') langName = 'python';
          if (langName === 'console' || langName === 'text' || langName === 'none') {
            langName = '';
          }
          let tempContent = this.uReplaceStr(this.$(el).text());
          if (tempContent) {
            content = `\n\`\`\` ${langName}\n${tempContent}\n\`\`\`\n`;
          }
        } else {
          content = this.getMarkdown(el);
        }
      }
      break;
      case 'OL': {
        let className = this.$(el).attr('class');
        let that = this;
        let templateContent = '';
        if (className === 'arabic simple') {
          let lis = this.$(el).find('li');
          lis.each(function (index, el) {
            templateContent = templateContent + `\n1. ${that.uReplaceStr(that.getPcontent(that.$(el)))}`;
          });
          content = templateContent + '\n';
        }
      }
      break;
      case 'UL': {
        let className = this.$(el).attr('class');
        let that = this;
        let templateContent = '';
        if (!className) {
          let lis = this.$(el).find('li');
          lis.each(function (index, el) {
            templateContent = templateContent + `\n- ${that.uReplaceStr(that.getMarkdown(el))}`;
          });
          content = templateContent + '\n';
        } else if (className.indexOf('simple') !== -1) {
          let lis = this.$(el).find('li');
          lis.each(function (index, el) {
            templateContent = templateContent + `\n- ${that.uReplaceStr(that.getPcontent(that.$(el)))}`;
          });
          content = templateContent + '\n';
        } else {
          let lis = this.$(el).find('li');
          lis.each(function (index, el) {
            templateContent = templateContent + `\n- ${that.uReplaceStr(that.getMarkdown(el))}`;
          });
          content = templateContent + '\n';
        }
      }
      break;
      case 'BLOCKQUOTE': {
        content = this.getMarkdown(el);
      }
      break;
      case 'DL': {
        content = this.getMarkdown(el);
      }
      break;
      case 'DT': {
        content = this.getPcontent(el);
      }
      break;
      case 'DD': {
        content = this.getMarkdown(el);
      }
      break;
      default:
        content = '';
        break;
      }
      return content;
    }

    getHtext(el) {
      let tempContent = this.replaceCode(this.$(el));
      let text = this.$(this.$(tempContent)).text();
      return text.replace('¶', '');
    }

    getPcontent(el) {
      let tempContent = this.replaceSpec(this.$(el));
      tempContent = this.replaceCode(this.$(el));
      tempContent = this.replaceA(this.$(tempContent));
      tempContent = this.replaceCite(this.$(tempContent));
      tempContent = this.replaceStrong(this.$(tempContent));
      tempContent = this.replaceSpan(this.$(tempContent));
      tempContent = this.replaceEm(this.$(tempContent));
      tempContent = this.$(tempContent).text();
      let content = `\n${tempContent}\n`;
      return content;
    }

    replaceCode($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      let codes = $el.find('code');
      if (codes.length) {
        codes.each(function (index, el) {
          let outHtml = that.$(el).prop("outerHTML");
          let text = that.$(el).text();
          hContent = hContent.replace(outHtml, `\`\`${text}\`\``);
        });
      }
      return hContent;
    }

    replaceA($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      let as = $el.find('a');
      if (as.length) {
        as.each(function (index, el) {
          let outHtml = that.$(el).prop("outerHTML");
          let href = that.$(el).attr('href');
          if (href.substring(0, 2) === '..') {
            href = that.baseUrl + href.substring(2, href.length);
          }
          let text = that.$(el).text();
          hContent = hContent.replace(outHtml, `[${text}](${href})`);
        });
      }
      return hContent;
    }

    replaceAForTable(content, $el) {
      let that = this;
      let as = $el.find('a');
      if (as.length) {
        as.each(function (index, el) {
          let href = that.$(el).attr('href');
          let text = that.$(el).text();
          if (href.substring(0, 2) === '..') {
            href = that.baseUrl + href.substring(2, href.length);
          }
          content = content.replace(text, `[${text}](${href})`);
        });
      }
      return content;
    }

    replaceSpan($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      let spans = $el.find('span');
      if (spans.length) {
        spans.each(function (index, el) {
          let outHtml = that.$(el).prop("outerHTML");
          let text = that.$(el).text();
          let className = that.$(el).attr('class');
          if (className === 'classifier' || className === 'versionmodified') {
            hContent = hContent.replace(outHtml, `*${that.uReplaceStr(text)}* `);
          } else {
            hContent = hContent.replace(outHtml, `${text}`);
          }
        });
      }
      return hContent;
    }

    replaceStrong($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      let strongs = $el.find('strong');
      if (strongs.length) {
        strongs.each(function (index, el) {
          let outHtml = that.$(el).prop("outerHTML");
          let text = that.$(el).text();
          hContent = hContent.replace(outHtml, `**${text}**`);
        });
      }
      return hContent;
    }

    replaceCite($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      let cites = $el.find('cite');
      if (cites.length) {
        cites.each(function (index, el) {
          let outHtml = that.$(el).prop("outerHTML");
          let text = that.$(el).text();
          hContent = hContent.replace(outHtml, `*${text}*`);
        });
      }
      return hContent;
    }

    replaceEm($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      let ems = $el.find('em');
      if (ems.length) {
        ems.each(function (index, el) {
          let outHtml = that.$(el).prop("outerHTML");
          let oText = that.$(el).text();
          let text = that.$(el).text();
          text = that.uReplaceStr(text);
          text = text.replace(/\*/g, '\\*');
          let bText = `*${text}*`;
          if (oText.match(/^\s/)) {
            bText = ' ' + bText;
          }
          if (oText.match(/\s$/)) {
            bText =  bText + ' ';
          }
          hContent = hContent.replace(outHtml, ' ' + bText + ' ');
        });
      }
      return hContent;
    }

    replaceSpec($el) {
      let that = this;
      let hContent = $el.prop("outerHTML");
      hContent = hContent.replace(/\*/g, '\\*');
      return hContent;
    }

    uReplaceStr(str) {
      return str.replace(/(^\n*)|(\n*$)/g, "").replace(/(^\s*)|(\s*$)/g, "");
    }
    
  }

  window.c = new Convert({
    baseUrl: 'https://numpy.org/devdocs',
    baseImgDir: '/static/images/',
    el: `#spc-section-body`,
    $: window.$
  });

  // c.getMarkdown();
  console.log(window.c.getMarkdown());
}
(function(selector) {
  let el = $(selector);
  let dtSet = el.find('dt');
  let ddSet = el.find('dd');
  let content = '';
  dtSet.each((index, el) => {
    let ddItem = ddSet[index];
    let dd_a_set = $(ddItem).find('a');
    let contentA = '';
    dd_a_set.each((idx, el_a) => {
      let href = $(el_a).attr('href');
      href = href.replace('..', '');
      contentA += `[${$(el_a).text()}](${href}), `;
    });
    content += `\n- **${$(el).text()}** - ${contentA}`;
  });
  console.log(content);
})('#d1');

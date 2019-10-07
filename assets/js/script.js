
$(document).ready(function() {

  /* make image paragraph styling possible */
  $("img").each(function() {
    var par = $(this).parent();
    if (par.is("p")) {
      par.before('<div class="clearfix"></div>');
      par.addClass('img');
    }
  });

  /* handle small/large image click event */
  $("p.img").on('click', function() {
    $(this).toggleClass("large-img");
  });

  /* add top-ul for CSS styling and "span" wrap top li texts */
  $("ul").each(function() {
    if (!$(this).parent().is("li")) {
      if ($(this).has("li").has("ul").length) {
        $(this).addClass("top-ul");
        $(this).children("li").each(function() {
          if (!$(this).contents().first().is("ul")) {
            $(this).contents().first().wrap('<span class="top-li"></span>');
          }
        });
      }
    }
  });
});


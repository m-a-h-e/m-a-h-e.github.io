
$(document).ready(function() {

  /* rearrange images and their caption and handle small/large click event */
  $("img").each(function() {
    if ($(this).parent().is("p")) {
      $(this).parent().before('<div class="clearfix"></div>');
      $(this).parent().wrap('<div class="img"></div>');
      $(this).insertBefore($(this).parent());
    } else {
      $(this).before('<div class="clearfix"></div>');
      $(this).wrap('<div class="img"></div>');
    }
  });
  $("div.img").on('click', function() {
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


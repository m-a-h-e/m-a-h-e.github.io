
$(document).ready(function() {
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

  $("ul").each(function() {
    if (!$(this).parent().is("li")) {
      if ($(this).has("li").has("ul").length) {
        $(this).addClass("top-ul");
        $(this).children("li").each(function() {
          //console.log($(this).text());
          // TODO wrap top-ul li text into <span>...</span> but without other (ul/li) sub-elements
        });
      }
    }
  });
});


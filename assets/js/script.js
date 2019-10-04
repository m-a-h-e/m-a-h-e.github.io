
$(document).ready(function() {
  $("img").each(function(){
    $(this).addClass("small");
    if ($(this).parent().is("p")) {
      $(this).parent().addClass("small");
      $(this).parent().addClass("caption");
      $(this).parent().before('<div class="clearfix"></div>');
    }
  });
  $("img").on('click', function(){
    $(this).toggleClass("small");
    if ($(this).parent().is("p")) {
      $(this).parent().toggleClass("small");
    }
  });
});


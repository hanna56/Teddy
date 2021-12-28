$(".loading").hide();
(function (){
    const wrap_container = document.querySelector('#response-container') ;
    const key = "9fPJKxAQCYzKDBVj50nwyT8tGZtyMYPD2JK5ivluMGE" ;
    const initr = wrap_container.innerHTML ;

    var page = 1 ;
    
    $('.keyword').click(function(){
        kywd = $(this).text()
        wrap_container.innerHTML = initr;
        $(".loading").show();
        doSeacrh(kywd);
    });
    
    
    
    
    
    function doSeacrh (kywd){
        var url = 'https://api.unsplash.com/search/photos?page='+page+'&query=';
        if( kywd != '') {
        url += kywd ;
        var ar = new XMLHttpRequest () ;
        ar.open('get', url, true);
        ar.setRequestHeader('Authorization', 'Client-ID '+ key);
        $(".loading").hide();
        ar.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
             let data = JSON.parse(this.responseText) ; 
             let tx = '';
    
             for (let i = 0; i < 10; i++)
             {
                tx = `<div>
                <figure>
                     <img src="${data.results[i].urls.small}" alt="${data.results[i].description}">
                     <figcaption>
                         <p>Description : ${data.results[i].description}</p>
                         <p>Upload by : ${data.results[i].user.name} (<em>@${data.results[i].user.username}</em>)</p>
                         <p><a href="${data.results[i].links.download}" target="_blank">Download</a></p>
                     </figcaption>
                 </figure>
                 </div>`;
                 wrap_container.insertAdjacentHTML('afterbegin',tx) ;
             }
             
            }
        };
        ar.send();
        }
    }
    })();
    
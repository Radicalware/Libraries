
#include "Server/Nix_Server.h"

#ifdef NIX_BASE

Nix_Server::Nix_Server()
{

}

void Nix_Server::listen(const xstring& port)
{
    if (port != "")
        m_port = port;

    auto doit = [](int connft){
        recv(connfd, &leng, sizeof(leng), 0);
        recv(connfd, rcvline, leng, 0);

        printf("From Client: %s\n", rcvline);

        strcpy(msg, "Server Recived Client Request\n");
        leng = strlen(msg);

        send(connfd, &leng, sizeof(leng), 0);
        send(connfd, msg, leng, 0);     
    };

    pid_t pid;
    int listenfd, connfd, bnd, lis, retVal;
    socklen_t len;

    struct sockaddr_in cliaddr, servaddr;

    listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if(listenfd < 0)
        perror("Socket Creation Error:");
    

    memset(&servaddr, 0, sizeof(servaddr));       // clear mem space
    servaddr.sin_family = AF_INET;                // set IPv4
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY); // accept from any address
    servaddr.sin_port = htons(m_port);            // set the port

    bnd = bind(listenfd, (struct sockaddr*) &servaddr, sizeof(servaddr));
    if(bnd < 0)
        perror("Bind Error:");
    
    lis = listen(listenfd, BACKLOG);
    if(lis < 0)
        perror("Listen Error:");
    
    while(true)
    {
        len = sizeof(cliaddr);
        connfd = accept(listenfd, (struct sockaddr*) &cliaddr, &len);
        if(connfd < 0)
            perror("Accept Error");

        if((pid = fork()) == 0)
        {
            retVal = close(listenfd);
            if(retVal < 0)
                perror("Close Error");
            
            doit(connfd);
            exit(0);
        }
        retVal = close(connfd);
        if(retVal < 0)
            perror("Close Error:");
        
        wait(nullptr); // Wait For the Child Process
    }
}


#endif // NIX_BASE
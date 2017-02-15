//================================
// 文字列のコード変換
//================================
#include"stdafx.h"

#include<Windows.h>
#include<WinNls.h>
#include"ConfigString.h"


//=========================================================================
int ShiftjisToUnicode(const char *buf, wchar_t *dest, int size)
{
	// 出力バッファに必要な長さを求める
	int orgLen = (int)strlen(buf);
	int newLen = MultiByteToWideChar(CP_ACP, 0, buf, orgLen, NULL, 0);

	if( newLen+1 > size ) return 1;		// バッファが足りない!!

	// SHIFT-JIS文字列をUNICODE文字列(BSTR)に変換
	int ret = MultiByteToWideChar(CP_ACP, 0, buf, orgLen, dest, newLen);
	dest[ret] = NULL;

	return 0;
}

//=========================================================================
//	@brief		Shift-jisの文字列を, UTF8文字列に変換する
//=========================================================================
std::wstring ShiftjisToUnicode(std::string buf)
{
	int size = buf.size() + 1;
	WCHAR* szBuf = new WCHAR[size];

	ShiftjisToUnicode(buf.c_str(), szBuf, size);
	std::wstring strBuf = szBuf;

	delete[] szBuf;

	return strBuf;
}



//=========================================================================
int UnicodeToShiftjis(const wchar_t *buf, char *dest, int size)
{
	// BSTRの長さを求める
	int orgLen = SysStringLen(SysAllocString(buf));

	// 全部マルチバイトになる場合の最大長+1(NULL文字分) のバッファを確保する
	int newSize = orgLen * 2 + 1;
	char *str = new char[newSize];

	// UNICODE文字列(BSTR)をSHIFT-JIS文字列に変換
	int ret = WideCharToMultiByte(CP_ACP, 0, buf, orgLen, str, newSize, NULL, NULL);
	str[ret] = NULL;			// NULL文字

	// 出力領域にコピー
	strncpy_s(dest, size, str, size);
	delete str;			 // 一時バッファ解放

	return 0;
}

//=========================================================================
/*!	@brief		Unicodeの文字列を, Shift-jis文字列に変換する
	*	@param[in]	buf  = 変換したいUnicodeの文字列
	*	@param[out]	dest = 変換後のShift-jisの文字列を格納するバッファへのポインタ
	*	@param[in]	size = 変換後の文字列を格納するバッファのサイズ
	*	@retval		0     = 成功
	*	@retval		0以外 = 失敗
	*/
//=========================================================================
std::string UnicodeToShiftjis(const std::wstring& strWBuf)
{
	int size = strWBuf.size()*2 + 1;
	char* szBuf = new char[size];

	UnicodeToShiftjis(strWBuf.c_str(), szBuf, size);
	std::string strBuf = szBuf;

	delete[] szBuf;

	return strBuf;
}

//=========================================================================
//	@brief		UTF8の文字列を, Shift-jis文字列に変換する
//=========================================================================
std::string UTF8toSJIS(const std::string& strBuf)
{
	BYTE* buffUtf8 = new BYTE[strBuf.size()+1];
	memset(buffUtf8, NULL, strBuf.size()+1);
	memcpy(buffUtf8, strBuf.c_str(), strBuf.size());

	//UTF-8からUTF-16へ変換
	const int nSize = ::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, NULL, 0 );

	BYTE* buffUtf16 = new BYTE[ nSize * 2 + 2 ];
	memset(buffUtf16, NULL, (nSize * 2 + 2));
	::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, (LPWSTR)buffUtf16, nSize );
	

	//UTF-16からShift-JISへ変換
	const int nSizeSJis = ::WideCharToMultiByte( CP_ACP, 0, (LPCWSTR)buffUtf16, -1, NULL, 0, NULL, NULL );


	BYTE* buffSJis = new BYTE[ nSizeSJis * 2 ];
	ZeroMemory( buffSJis, nSizeSJis * 2 );
	::WideCharToMultiByte( CP_ACP, 0, (LPCWSTR)buffUtf16, -1, (LPSTR)buffSJis, nSizeSJis, NULL, NULL );

	std::string strSJIS = (char*)buffSJis;

	delete[] buffUtf16;
	delete[] buffSJis;
	delete[] buffUtf8;

	return strSJIS;
}

//=========================================================================
//	@brief		UTF8の文字列を, Unicode文字列に変換する
//=========================================================================
std::wstring UTF8toUnicode(const std::string& strBuf)
{
	BYTE* buffUtf8 = new BYTE[strBuf.size()+1];
	memset(buffUtf8, NULL, strBuf.size()+1);
	memcpy(buffUtf8, strBuf.c_str(), strBuf.size());

	//UTF-8からUTF-16へ変換
	const int nSize = ::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, NULL, 0 );

	wchar_t* buffUtf16 = new wchar_t[ nSize + 1 ];
	memset(buffUtf16, NULL, (nSize + 1));
	::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, (LPWSTR)buffUtf16, nSize );
	

	std::wstring strUTF16 = buffUtf16;

	delete[] buffUtf16;
	delete[] buffUtf8;

	return strUTF16;
}
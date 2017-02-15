//================================
// ������̃R�[�h�ϊ�
//================================
#include"stdafx.h"

#include<Windows.h>
#include<WinNls.h>
#include"ConfigString.h"


//=========================================================================
int ShiftjisToUnicode(const char *buf, wchar_t *dest, int size)
{
	// �o�̓o�b�t�@�ɕK�v�Ȓ��������߂�
	int orgLen = (int)strlen(buf);
	int newLen = MultiByteToWideChar(CP_ACP, 0, buf, orgLen, NULL, 0);

	if( newLen+1 > size ) return 1;		// �o�b�t�@������Ȃ�!!

	// SHIFT-JIS�������UNICODE������(BSTR)�ɕϊ�
	int ret = MultiByteToWideChar(CP_ACP, 0, buf, orgLen, dest, newLen);
	dest[ret] = NULL;

	return 0;
}

//=========================================================================
//	@brief		Shift-jis�̕������, UTF8������ɕϊ�����
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
	// BSTR�̒��������߂�
	int orgLen = SysStringLen(SysAllocString(buf));

	// �S���}���`�o�C�g�ɂȂ�ꍇ�̍ő咷+1(NULL������) �̃o�b�t�@���m�ۂ���
	int newSize = orgLen * 2 + 1;
	char *str = new char[newSize];

	// UNICODE������(BSTR)��SHIFT-JIS������ɕϊ�
	int ret = WideCharToMultiByte(CP_ACP, 0, buf, orgLen, str, newSize, NULL, NULL);
	str[ret] = NULL;			// NULL����

	// �o�͗̈�ɃR�s�[
	strncpy_s(dest, size, str, size);
	delete str;			 // �ꎞ�o�b�t�@���

	return 0;
}

//=========================================================================
/*!	@brief		Unicode�̕������, Shift-jis������ɕϊ�����
	*	@param[in]	buf  = �ϊ�������Unicode�̕�����
	*	@param[out]	dest = �ϊ����Shift-jis�̕�������i�[����o�b�t�@�ւ̃|�C���^
	*	@param[in]	size = �ϊ���̕�������i�[����o�b�t�@�̃T�C�Y
	*	@retval		0     = ����
	*	@retval		0�ȊO = ���s
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
//	@brief		UTF8�̕������, Shift-jis������ɕϊ�����
//=========================================================================
std::string UTF8toSJIS(const std::string& strBuf)
{
	BYTE* buffUtf8 = new BYTE[strBuf.size()+1];
	memset(buffUtf8, NULL, strBuf.size()+1);
	memcpy(buffUtf8, strBuf.c_str(), strBuf.size());

	//UTF-8����UTF-16�֕ϊ�
	const int nSize = ::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, NULL, 0 );

	BYTE* buffUtf16 = new BYTE[ nSize * 2 + 2 ];
	memset(buffUtf16, NULL, (nSize * 2 + 2));
	::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, (LPWSTR)buffUtf16, nSize );
	

	//UTF-16����Shift-JIS�֕ϊ�
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
//	@brief		UTF8�̕������, Unicode������ɕϊ�����
//=========================================================================
std::wstring UTF8toUnicode(const std::string& strBuf)
{
	BYTE* buffUtf8 = new BYTE[strBuf.size()+1];
	memset(buffUtf8, NULL, strBuf.size()+1);
	memcpy(buffUtf8, strBuf.c_str(), strBuf.size());

	//UTF-8����UTF-16�֕ϊ�
	const int nSize = ::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, NULL, 0 );

	wchar_t* buffUtf16 = new wchar_t[ nSize + 1 ];
	memset(buffUtf16, NULL, (nSize + 1));
	::MultiByteToWideChar( CP_UTF8, 0, (LPCSTR)buffUtf8, -1, (LPWSTR)buffUtf16, nSize );
	

	std::wstring strUTF16 = buffUtf16;

	delete[] buffUtf16;
	delete[] buffUtf8;

	return strUTF16;
}
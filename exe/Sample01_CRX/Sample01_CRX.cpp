//=============================================
// �N���W�b�g�J�[�h�F�؂̃f�[�^��p���������T���v��
// �Q�lURL�F
// �EDropout�F�f�B�[�v���[�j���O�̉Εt�����A�P���ȕ��@�ŉߊw�K��h��
//	https://deepage.net/deep_learning/2016/10/17/deeplearning_dropout.html
//
// �T���v���f�[�^URL:
// https://archive.ics.uci.edu/ml/datasets/Credit+Approval
//  �f�[�^�{��
//		Data Folder > crx.data
//  �f�[�^�t�H�[�}�b�g�ɂ���
//		Data Folder > crx.names
//=============================================


#include "stdafx.h"

#include <boost/tokenizer.hpp>
#include<boost/algorithm/string.hpp>


#include"Library/DataFormat/DataFormatStringArray/DataFormat.h"


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif

	// �t�H�[�}�b�g��ǂݍ���
	auto pDataFormat = Gravisbell::DataFormat::StringArray::CreateDataFormatFromXML(L"DataFormat.xml");
	if(pDataFormat == NULL)
		return -1;

	// CSV�t�@�C����ǂݍ���Ńt�H�[�}�b�g�ɒǉ�
	{
		// �t�@�C���I�[�v��
		FILE* fp = fopen("../../SampleData/crx.csv", "r");
		if(fp == NULL)
		{
			delete pDataFormat;
			return -1;
		}

		wchar_t szBuf[1024];
		while(fgetws(szBuf, sizeof(szBuf)/sizeof(wchar_t)-1, fp))
		{
			size_t len = wcslen(szBuf);
			if(szBuf[len-1] == '\n')
				szBuf[len-1] = NULL;

			// ","(�J���})��؂�ŕ���
			std::vector<std::wstring> lpBuf;
			boost::split(lpBuf, szBuf, boost::is_any_of(L","));

			std::vector<const wchar_t*> lpBufPointer;
			for(auto& buf : lpBuf)
				lpBufPointer.push_back(buf.c_str());


			pDataFormat->AddDataByStringArray(&lpBufPointer[0]);
		}

		// �t�@�C���N���[�Y
		fclose(fp);
	}


	// ���K��
	pDataFormat->Normalize();


	// �t�H�[�}�b�g���폜
	delete pDataFormat;


	return 0;
}


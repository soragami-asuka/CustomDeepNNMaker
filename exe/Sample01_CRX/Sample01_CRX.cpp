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
#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Layer/IOData/IODataLayer/IODataLayer.h"

using namespace Gravisbell;

/** �T���v���f�[�^�̓ǂݍ��� */
DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath);
/** ���C���[DLL�Ǘ��N���X�̍쐬 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void);

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif


	// �T���v���f�[�^�̓ǂݍ���
	Gravisbell::DataFormat::IDataFormatBase* pSampleData = ::LoadSampleData(L"DataFormat.xml", L"../../SampleData/crx.csv");

	// ���C���[DLL�Ǘ��N���X���쐬
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = ::CreateLayerDLLManager();
	if(pLayerDLLManager == NULL)
	{
		delete pSampleData;
	}

	// ���o�͐M�����C���[���쐬
	Layer::IOData::IIODataLayer* pInputLayer  = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"input") );	// ���͐M��
	Layer::IOData::IIODataLayer* pOutputLayer = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"output") );	// �o�͐M��
	for(U32 dataNum=0; dataNum<pSampleData->GetDataCount(); dataNum++)
	{
		pInputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"input"));
		pOutputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"output"));
	}

	// �������J��
	delete pInputLayer;
	delete pOutputLayer;
	delete pLayerDLLManager;
	delete pSampleData;

	return 0;
}


/** �T���v���f�[�^�̓ǂݍ��� */
Gravisbell::DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath)
{
	// �t�H�[�}�b�g��ǂݍ���
	auto pDataFormat = Gravisbell::DataFormat::StringArray::CreateDataFormatFromXML(formatFilePath.c_str());
	if(pDataFormat == NULL)
		return NULL;

	// CSV�t�@�C����ǂݍ���Ńt�H�[�}�b�g�ɒǉ�
	{
		// �t�@�C���I�[�v��
		FILE* fp = _wfopen(dataFilePath.c_str(), L"r");
		if(fp == NULL)
		{
			delete pDataFormat;
			return NULL;
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

	return pDataFormat;
}
/** ���C���[DLL�Ǘ��N���X�̍쐬 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void)
{
	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	// DLL�̓ǂݍ���.
	// FullyConnect
	{
	#ifdef _DEBUG
		if(pDLLManager->ReadLayerDLL(L"../../Debug/Gravisbell.Layer.NeuralNetwork.FullyConnect_Activation.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#else
		if(pDLLManager->ReadLayerDLL(L"../../Release/Gravisbell.Layer.NeuralNetwork.FullyConnect_Activation.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#endif
		{
			delete pDLLManager;
			return NULL;
		}
	}
	// Feedforward
	{
	#ifdef _DEBUG
		if(pDLLManager->ReadLayerDLL(L"../../Debug/Gravisbell.Layer.NeuralNetwork.FeedforwardNeuralNetwork.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#else
		if(pDLLManager->ReadLayerDLL(L"../../Release/Gravisbell.Layer.NeuralNetwork.FeedforwardNeuralNetwork.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#endif
		{
			delete pDLLManager;
			return NULL;
		}
	}

	return pDLLManager;
}

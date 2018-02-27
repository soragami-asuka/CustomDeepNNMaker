/*--------------------------------------------
 * FileName  : Convolution_FUNC.cpp
 * LayerName : ��݂��݃j���[�����l�b�g���[�N
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : ��݂��݃j���[�����l�b�g���[�N.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"Convolution_FUNC.hpp"


// {F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA}
static const Gravisbell::GUID g_guid(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 



struct StringData
{
    std::wstring name;
    std::wstring text;
};

namespace DefaultLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = L"ja";

    /** Base */
    static const StringData g_baseData = 
    {
        L"��݂��݃j���[�����l�b�g���[�N",
        L"��݂��݃j���[�����l�b�g���[�N."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"FilterSize",
            {
                L"�t�B���^�T�C�Y",
                L"��݂��݂��s�����͐M����",
            }
        },
        {
            L"Input_Channel",
            {
                L"���̓`�����l����",
                L"���̓`�����l����",
            }
        },
        {
            L"Output_Channel",
            {
                L"�o�̓`�����l����",
                L"�o�͂����`�����l���̐�",
            }
        },
        {
            L"Stride",
            {
                L"�t�B���^�ړ���",
                L"��݂��݂��ƂɈړ�����t�B���^�̈ړ���",
            }
        },
        {
            L"Padding",
            {
                L"�p�f�B���O�T�C�Y",
                L"",
            }
        },
        {
            L"PaddingType",
            {
                L"�p�f�B���O���",
                L"�p�f�B���O���s���ۂ̕��@�ݒ�",
            }
        },
        {
            L"Initializer",
            {
                L"�������֐�",
                L"�������֐��̎��",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PaddingType",
            {
                {
                    L"zero",
                    {
                        L"�[���p�f�B���O",
                        L"�s������0�Ŗ��߂�",
                    },
                },
            }
        },
    };



    /** ItemData Runtiime Parameter <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Runtime = 
    {
        {
            L"UpdateWeigthWithOutputVariance",
            {
                L"�o�͂̕��U��p���ďd�݂��X�V����t���O",
                L"�o�͂̕��U��p���ďd�݂��X�V����t���O.true�ɂ����ꍇCalculate���ɏo�͂̕��U��1�ɂȂ�܂ŏd�݂��X�V����.",
            }
        },
    };


    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Runtime =
    {
    };


}


namespace CurrentLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;


    /** Base */
    static StringData g_baseData = DefaultLanguage::g_baseData;


    /** ItemData Layer Structure <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;


    /** ItemData Runtime <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Runtime;

    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Runtime;

}



/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : �t�B���^�T�C�Y
	  * ID   : FilterSize
	  * Text : ��݂��݂��s�����͐M����
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : ���̓`�����l����
	  * ID   : Input_Channel
	  * Text : ���̓`�����l����
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Input_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Input_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Input_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : �o�̓`�����l����
	  * ID   : Output_Channel
	  * Text : �o�͂����`�����l���̐�
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Output_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : �t�B���^�ړ���
	  * ID   : Stride
	  * Text : ��݂��݂��ƂɈړ�����t�B���^�̈ړ���
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : �p�f�B���O�T�C�Y
	  * ID   : Padding
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"Padding",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Padding"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : �p�f�B���O���
	  * ID   : PaddingType
	  * Text : �p�f�B���O���s���ۂ̕��@�ݒ�
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PaddingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"zero",
			L"�[���p�f�B���O",
			L"�s������0�Ŗ��߂�");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	/** Name : �������֐�
	  * ID   : Initializer
	  * Text : �������֐��̎��
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_String(
			L"Initializer",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Initializer"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Initializer"].text.c_str(),
			L"glorot_uniform"));

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a runtime parameters.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameter(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : �o�͂̕��U��p���ďd�݂��X�V����t���O
	  * ID   : UpdateWeigthWithOutputVariance
	  * Text : �o�͂̕��U��p���ďd�݂��X�V����t���O.true�ɂ����ꍇCalculate���ɏo�͂̕��U��1�ɂȂ�܂ŏd�݂��X�V����.
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Bool(
			L"UpdateWeigthWithOutputVariance",
			CurrentLanguage::g_lpItemData_Learn[L"UpdateWeigthWithOutputVariance"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"UpdateWeigthWithOutputVariance"].text.c_str(),
			false));

	return pLayerConfig;
}

/** Create runtime parameter from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateRuntimeParameter();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}



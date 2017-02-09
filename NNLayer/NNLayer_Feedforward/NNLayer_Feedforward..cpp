// NNLayer_Feedforward.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "NNLayer_Feedforward.h"

#include"LayerErrorCode.h"
#include"NNLayerConfig.h"


/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetLayerCode(GUID& o_layerCode)
{
	// {BEBA34EC-C30C-4565-9386-56088981D2D7}
	static const GUID layerCode = { 0xbeba34ec, 0xc30c, 0x4565, { 0x93, 0x86, 0x56, 0x8, 0x89, 0x81, 0xd2, 0xd7 } };

	o_layerCode = layerCode;

	return CustomDeepNNLibrary::LAYER_ERROR_NONE;
}
/** �o�[�W�����R�[�h���擾����.
	@param o_versionCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode)
{
	static const CustomDeepNNLibrary::VersionCode versionCode(0, 0, 0, 0);

	o_versionCode = versionCode;

	return CustomDeepNNLibrary::LAYER_ERROR_NONE;
}


/** ���C���[�ݒ���쐬���� */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfig(void)
{
	GUID layerCode;
	GetLayerCode(layerCode);

	CustomDeepNNLibrary::VersionCode versionCode;
	GetVersionCode(versionCode);


	//=================================
	// ��̐ݒ�����쐬����
	//=================================
	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = CustomDeepNNLibrary::CreateEmptyLayerConfig(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	//=================================
	// �e�ݒ�A�C�e����ǉ����Ă���
	//=================================
	// �j���[������
	{
		CustomDeepNNLibrary::INNLayerConfigItem_Int* pItem = CustomDeepNNLibrary::CreateLayerCofigItem_Int("�j���[������", 0, 65535, 200);

		pLayerConfig->AddItem(pItem);
	}


	return pLayerConfig;
}

/** ���C���[�ݒ���쐬����
	@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
	@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
	@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
	@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = (CustomDeepNNLibrary::INNLayerConfigEx*)CreateLayerConfig();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}

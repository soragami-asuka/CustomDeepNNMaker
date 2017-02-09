//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#define EXPORT_API extern "C" __declspec(dllexport)


#include"LayerErrorCode.h"
#include"INNLayerConfig.h"
#include"INNLayer.h"

#include<guiddef.h>

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetLayerCode(GUID& o_layerCode);

/** �o�[�W�����R�[�h���擾����.
	@param o_versionCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode);


/** ���C���[�ݒ���쐬���� */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfig(void);
/** ���C���[�ݒ���쐬����
	@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
	@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
	@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
	@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


/** CPU�����p�̃��C���[���쐬.
	@param guid	�쐬���郌�C���[��GUID */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerCPU(GUID guid);

/** GPU�����p�̃��C���[���쐬
	@param guid �쐬���郌�C���[��GUID */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerGPU(GUID guid);

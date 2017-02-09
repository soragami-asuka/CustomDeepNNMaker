//=======================================
// ���C���[DLL�N���X
//=======================================
#ifndef __I_NN_LAYER_DLL_H__
#define __I_NN_LAYER_DLL_H__

#include<guiddef.h>

#include"INNLayer.h"

namespace CustomDeepNNLibrary
{
	class INNLayerDLL
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerDLL(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayerDLL(){}

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_layerCode)const = 0;
		/** �o�[�W�����R�[�h���擾����.
			@param o_versionCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode)const = 0;


		/** ���C���[�ݒ���쐬���� */
		virtual INNLayerConfig* CreateLayerConfig(void)const = 0;
		/** ���C���[�ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual INNLayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const = 0;

		
		/** CPU�����p�̃��C���[���쐬.
			GUID�͎������蓖��. */
		virtual INNLayer* CreateLayerCPU()const = 0;
		/** CPU�����p�̃��C���[���쐬
			@param guid	�쐬���C���[��GUID */
		virtual INNLayer* CreateLayerCPU(GUID guid)const = 0;
		
		/** GPU�����p�̃��C���[���쐬.
			GUID�͎������蓖��. */
		virtual INNLayer* CreateLayerGPU()const = 0;
		/** GPU�����p�̃��C���[���쐬 */
		virtual INNLayer* CreateLayerGPU(GUID guid)const = 0;
	};
}

#endif